# Persona Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-stage checkpointed pipeline that collects Amazon reviews, extracts behavioral signals via DeepSeek V3.2 on OpenRouter, auto-clusters them, and exports a college student persona as both JSON and Markdown.

**Architecture:** Staged pipeline in `pipeline/` with one file per stage; each stage reads a checkpoint file and writes its own, enabling crash recovery and isolated re-runs. All LLM calls route through a single `engine/llm.py` client factory using the OpenAI SDK pointed at OpenRouter. Existing `main.py` agent runner is untouched.

**Tech Stack:** Python 3.9, OpenAI SDK (OpenRouter-compatible), sentence-transformers, scikit-learn, Playwright, HuggingFace datasets, click, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `engine/llm.py` | Create | OpenRouter/DeepSeek client factory |
| `engine/extract.py` | Modify | Swap Anthropic → OpenAI-compatible calls |
| `engine/knowledge.py` | Modify | Swap Anthropic → OpenAI-compatible calls |
| `engine/persona_builder.py` | Modify | Swap Anthropic → OpenAI-compatible calls |
| `pipeline/__init__.py` | Create | Empty package marker |
| `pipeline/collect.py` | Create | Stage 1: HuggingFace pull + Playwright scraper |
| `pipeline/extract.py` | Create | Stage 2: batched signal extraction with resume |
| `pipeline/cluster.py` | Create | Stage 3: embed + auto-optimized KMeans |
| `pipeline/export.py` | Create | Stage 4: trait extraction, persona synthesis, dual output |
| `pipeline.py` | Create | CLI entry point (click group) |
| `config.yaml` | Modify | Add openrouter, scraper, clustering blocks |
| `.env.example` | Modify | Swap ANTHROPIC_API_KEY → OPENROUTER_API_KEY |
| `requirements.txt` | Modify | Add openai package |
| `tests/__init__.py` | Create | Empty |
| `tests/pipeline/__init__.py` | Create | Empty |
| `tests/test_llm.py` | Create | Unit tests for client factory |
| `tests/pipeline/test_collect.py` | Create | Unit tests for dedup logic |
| `tests/pipeline/test_extract.py` | Create | Unit tests for resume logic |
| `tests/pipeline/test_cluster.py` | Create | Unit tests for scoring + text flattening |
| `tests/pipeline/test_export.py` | Create | Unit tests for markdown generation + RAG index |

---

## Task 1: LLM Client Factory + Environment Setup

**Files:**
- Create: `engine/llm.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `config.yaml`
- Create: `tests/test_llm.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import os
import pytest


def test_get_client_returns_openai_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from engine.llm import get_client
    from openai import OpenAI
    client = get_client()
    assert isinstance(client, OpenAI)
    assert "openrouter" in str(client.base_url)


def test_model_constant():
    from engine.llm import MODEL
    assert MODEL == "deepseek/deepseek-v3.2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/test_llm.py -v
```
Expected: `ModuleNotFoundError: No module named 'engine.llm'`

- [ ] **Step 3: Add openai to requirements.txt**

Open `requirements.txt` and add after the `anthropic` line:
```
openai>=1.0.0
```

- [ ] **Step 4: Install openai**

```bash
cd /Users/adit/proxi-ai && .venv/bin/pip install openai
```

- [ ] **Step 5: Create `engine/llm.py`**

```python
"""OpenRouter LLM client factory.

All pipeline stages and engine modules import get_client() from here
instead of instantiating anthropic.Anthropic() directly.
"""
from __future__ import annotations

import os

from openai import OpenAI

MODEL: str = "deepseek/deepseek-v3.2"


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
```

- [ ] **Step 6: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 7: Run test to verify it passes**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/test_llm.py -v
```
Expected: `2 passed`

- [ ] **Step 8: Update `.env.example`**

Replace the contents of `.env.example`:
```
OPENROUTER_API_KEY=your-openrouter-key-here
```

- [ ] **Step 9: Update `config.yaml`** — add these blocks at the top (before `anthropic:`):

```yaml
openrouter:
  model: "deepseek/deepseek-v3.2"

scraper:
  target_queries:
    - "college backpack"
    - "student ID wallet"
    - "laptop stand desk"
    - "college dorm supplies"
    - "noise cancelling headphones student"
  max_reviews_per_query: 200
  headless: true

clustering:
  sweep_range: [3, 10]
  scoring_weights:
    silhouette: 0.5
    intra_similarity: 0.3
    inter_distance: 0.2

# Phase 2 — uncomment to activate
# active_persona: college_tech_entrepreneur
# scraper:
#   target_queries:
#     - "vercel review developer"
#     - "supabase postgres developer"
#     - "stripe payments startup"
#   g2_targets:
#     - "vercel"
#     - "supabase"
#     - "linear"
```

Also add to the `data:` section in `config.yaml`:
```yaml
  amazon_categories:
    - "raw_review_Electronics"
    - "raw_review_Software"
    - "raw_review_Office_Products"
    - "raw_review_Computers"
    - "raw_review_Clothing_Shoes_and_Jewelry"
    - "raw_review_Sports_and_Outdoors"
```

- [ ] **Step 10: Commit**

```bash
cd /Users/adit/proxi-ai && git add engine/llm.py requirements.txt .env.example config.yaml tests/__init__.py tests/test_llm.py
git commit -m "feat: add OpenRouter LLM client factory and env config"
```

---

## Task 2: Swap engine/extract.py to OpenRouter

**Files:**
- Modify: `engine/extract.py`

The file uses `anthropic.Anthropic` client and `client.messages.create(...)`. Replace with the OpenAI-compatible surface.

- [ ] **Step 1: Replace `extract_signals_batch` in `engine/extract.py`**

Replace the entire `extract_signals_batch` function (lines 123–182) with:

```python
def extract_signals_batch(
    reviews: list[dict],
    category: str,
    client,
) -> dict:
    """Extract behavioral signals from a single batch of reviews."""
    from engine.llm import MODEL
    import openai

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
```

Also remove `import anthropic` from the top of the file and remove the `MODEL: str = "claude-haiku-4-5-20251001"` constant (line 30) since MODEL now comes from `engine/llm.py`.

Also update `extract_all_signals` — remove the `client = anthropic.Anthropic()` default construction and replace with:

```python
    if client is None:
        from engine.llm import get_client
        client = get_client()
```

- [ ] **Step 2: Verify the smoke test still runs (dry)**

```bash
cd /Users/adit/proxi-ai && python -c "from engine.extract import extract_signals_batch, extract_all_signals; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/adit/proxi-ai && git add engine/extract.py
git commit -m "feat: swap engine/extract.py to OpenRouter/DeepSeek"
```

---

## Task 3: Swap engine/knowledge.py to OpenRouter

**Files:**
- Modify: `engine/knowledge.py`

- [ ] **Step 1: Update imports and client construction in `engine/knowledge.py`**

Remove `import anthropic` from the top.

In `train_persona_on_reviews` (line 198), replace:
```python
    if client is None:
        client = anthropic.Anthropic()
```
with:
```python
    if client is None:
        from engine.llm import get_client
        client = get_client()
```

Replace the `client.messages.create(...)` call (lines 250–255) with:
```python
    from engine.llm import MODEL
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
```

Replace the response parsing line:
```python
    response_text = response.content[0].text
```
with:
```python
    response_text = response.choices[0].message.content
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/adit/proxi-ai && python -c "from engine.knowledge import train_persona_on_reviews, load_persona_config; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/adit/proxi-ai && git add engine/knowledge.py
git commit -m "feat: swap engine/knowledge.py to OpenRouter/DeepSeek"
```

---

## Task 4: Swap engine/persona_builder.py to OpenRouter

**Files:**
- Modify: `engine/persona_builder.py`

- [ ] **Step 1: Update `synthesize_persona` in `engine/persona_builder.py`**

Remove `import anthropic` from the top.

Replace the `client.messages.create(...)` block (lines 122–128) with:
```python
    from engine.llm import MODEL
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw_text = response.choices[0].message.content.strip()
```

In `build_personas_from_signals` (line 171), replace:
```python
    if client is None:
        client = anthropic.Anthropic()
```
with:
```python
    if client is None:
        from engine.llm import get_client
        client = get_client()
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/adit/proxi-ai && python -c "from engine.persona_builder import synthesize_persona; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/adit/proxi-ai && git add engine/persona_builder.py
git commit -m "feat: swap engine/persona_builder.py to OpenRouter/DeepSeek"
```

---

## Task 5: Stage 1 — Review Collection (HuggingFace + Playwright)

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/collect.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_collect.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_collect.py
import json
from pathlib import Path
import pytest


def test_review_hash_is_deterministic():
    from pipeline.collect import _review_hash
    h1 = _review_hash("B001234567", "user42")
    h2 = _review_hash("B001234567", "user42")
    assert h1 == h2


def test_review_hash_differs_for_different_inputs():
    from pipeline.collect import _review_hash
    h1 = _review_hash("B001234567", "user42")
    h2 = _review_hash("B001234567", "user99")
    assert h1 != h2


def test_load_existing_hashes_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.collect.REVIEWS_PATH", tmp_path / "reviews.jsonl")
    from pipeline.collect import load_existing_hashes
    assert load_existing_hashes() == set()


def test_load_existing_hashes_reads_written_reviews(tmp_path, monkeypatch):
    reviews_path = tmp_path / "reviews.jsonl"
    reviews_path.write_text(
        json.dumps({"asin": "B001", "reviewer_id": "u1", "text": "good"}) + "\n" +
        json.dumps({"asin": "B002", "reviewer_id": "u2", "text": "bad"}) + "\n"
    )
    monkeypatch.setattr("pipeline.collect.REVIEWS_PATH", reviews_path)
    from pipeline.collect import load_existing_hashes, _review_hash
    hashes = load_existing_hashes()
    assert _review_hash("B001", "u1") in hashes
    assert _review_hash("B002", "u2") in hashes
    assert len(hashes) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_collect.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.collect'`

- [ ] **Step 3: Create `pipeline/__init__.py` and `tests/pipeline/__init__.py`**

Both are empty files.

- [ ] **Step 4: Create `pipeline/collect.py`**

```python
"""Stage 1: Review collection — HuggingFace dataset pull + Playwright gap-fill."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path

RAW_DIR = Path("data/raw")
REVIEWS_PATH = RAW_DIR / "reviews.jsonl"
ERRORS_PATH = RAW_DIR / "scrape_errors.jsonl"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _review_hash(asin: str, reviewer_id: str) -> str:
    return hashlib.md5(f"{asin}:{reviewer_id}".encode()).hexdigest()


def load_existing_hashes() -> set[str]:
    if not REVIEWS_PATH.exists():
        return set()
    hashes: set[str] = set()
    with open(REVIEWS_PATH) as f:
        for line in f:
            r = json.loads(line)
            hashes.add(_review_hash(r.get("asin", ""), r.get("reviewer_id", "")))
    return hashes


def _log_scrape_error(asin: str, reason: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_PATH, "a") as f:
        f.write(json.dumps({"asin": asin, "reason": reason}) + "\n")


# ---------------------------------------------------------------------------
# HuggingFace collection
# ---------------------------------------------------------------------------

def collect_huggingface(config: dict, existing_hashes: set[str]) -> int:
    """Pull reviews from McAuley-Lab/Amazon-Reviews-2023."""
    from datasets import load_dataset

    categories = config["data"]["amazon_categories"]
    max_per = config["data"]["max_reviews_per_category"]
    min_len = config["data"]["min_review_length"]
    verified_only = config["data"].get("verified_only", True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(REVIEWS_PATH, "a") as out:
        for category in categories:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                category,
                split="full",
                trust_remote_code=True,
            )
            written = 0
            for row in ds:
                if written >= max_per:
                    break
                if verified_only and not row.get("verified_purchase", False):
                    continue
                text = row.get("text", "") or ""
                if len(text.strip()) < min_len:
                    continue
                asin = row.get("asin", "")
                reviewer_id = row.get("user_id", row.get("reviewer_id", ""))
                h = _review_hash(asin, reviewer_id)
                if h in existing_hashes:
                    continue
                existing_hashes.add(h)
                out.write(json.dumps({
                    "source": "huggingface",
                    "category": category,
                    "rating": float(row.get("rating", row.get("overall", 0))),
                    "text": text,
                    "timestamp": str(row.get("timestamp", "")),
                    "asin": asin,
                    "reviewer_id": reviewer_id,
                }) + "\n")
                written += 1
                count += 1

    return count


# ---------------------------------------------------------------------------
# Playwright scraper (gap-fill)
# ---------------------------------------------------------------------------

async def scrape_amazon(config: dict, existing_hashes: set[str]) -> int:
    """Gap-fill with targeted Amazon scraping via Playwright."""
    from playwright.async_api import async_playwright

    queries = config["scraper"]["target_queries"]
    max_per_query = config["scraper"]["max_reviews_per_query"]
    headless = config["scraper"].get("headless", True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        for query in queries:
            asins = await _search_amazon_asins(page, query)
            for asin in asins[:5]:
                try:
                    reviews = await _scrape_asin_reviews(page, asin, max_per_query)
                    with open(REVIEWS_PATH, "a") as out:
                        for rev in reviews:
                            h = _review_hash(asin, rev.get("reviewer_id", ""))
                            if h in existing_hashes:
                                continue
                            existing_hashes.add(h)
                            rev["asin"] = asin
                            rev["source"] = "playwright"
                            out.write(json.dumps(rev) + "\n")
                            count += 1
                except Exception as e:
                    _log_scrape_error(asin, str(e))

                time.sleep(random.uniform(1.5, 3.5))

        await browser.close()

    return count


async def _search_amazon_asins(page, query: str) -> list[str]:
    await page.goto(f"https://www.amazon.com/s?k={query.replace(' ', '+')}")
    await page.wait_for_timeout(2000)

    if "captcha" in page.url.lower() or await page.query_selector('[action*="captcha"]'):
        _log_scrape_error("search", f"CAPTCHA on query: {query}")
        return []

    asins: list[str] = []
    items = await page.query_selector_all('[data-asin]')
    for item in items:
        asin = await item.get_attribute('data-asin')
        if asin and len(asin) == 10:
            asins.append(asin)
    return asins[:5]


async def _scrape_asin_reviews(page, asin: str, max_reviews: int) -> list[dict]:
    reviews: list[dict] = []
    page_num = 1

    while len(reviews) < max_reviews:
        url = f"https://www.amazon.com/product-reviews/{asin}?pageNumber={page_num}"
        response = await page.goto(url)

        if not response or response.status != 200:
            _log_scrape_error(asin, f"HTTP {response.status if response else 'no response'}")
            break

        if "captcha" in page.url.lower():
            _log_scrape_error(asin, "CAPTCHA on reviews page")
            break

        await page.wait_for_timeout(1500)
        review_elements = await page.query_selector_all('[data-hook="review"]')
        if not review_elements:
            break

        for el in review_elements:
            if len(reviews) >= max_reviews:
                break
            text_el = await el.query_selector('[data-hook="review-body"]')
            rating_el = await el.query_selector('[data-hook="review-star-rating"]')
            reviewer_el = await el.query_selector('.a-profile-name')

            text = await text_el.inner_text() if text_el else ""
            rating_str = await rating_el.get_attribute('class') if rating_el else ""
            reviewer = await reviewer_el.inner_text() if reviewer_el else f"reviewer_{page_num}"

            rating_match = re.search(r'a-star-(\d)', rating_str)
            rating = int(rating_match.group(1)) if rating_match else 3

            if len(text.strip()) > 20:
                reviews.append({
                    "text": text.strip(),
                    "rating": float(rating),
                    "reviewer_id": reviewer.strip(),
                    "category": "scraped",
                })

        page_num += 1

    return reviews
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_collect.py -v
```
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
cd /Users/adit/proxi-ai && git add pipeline/__init__.py pipeline/collect.py tests/pipeline/__init__.py tests/pipeline/test_collect.py
git commit -m "feat: add Stage 1 review collection (HuggingFace + Playwright)"
```

---

## Task 6: Stage 2 — Signal Extraction Runner

**Files:**
- Create: `pipeline/extract.py`
- Create: `tests/pipeline/test_extract.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_extract.py
import json
from pathlib import Path
import pytest


def test_count_completed_batches_zero_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.extract.SIGNALS_PATH", tmp_path / "signals.jsonl")
    from pipeline.extract import _count_completed_batches
    assert _count_completed_batches() == 0


def test_count_completed_batches_counts_lines(tmp_path, monkeypatch):
    signals_path = tmp_path / "signals.jsonl"
    signals_path.write_text(
        json.dumps({"pain_points": []}) + "\n" +
        json.dumps({"pain_points": []}) + "\n"
    )
    monkeypatch.setattr("pipeline.extract.SIGNALS_PATH", signals_path)
    from pipeline.extract import _count_completed_batches
    assert _count_completed_batches() == 2


def test_build_batches():
    from pipeline.extract import _build_batches
    reviews = [{"text": f"review {i}"} for i in range(75)]
    batches = _build_batches(reviews, batch_size=30)
    assert len(batches) == 3
    assert len(batches[0]) == 30
    assert len(batches[2]) == 15
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_extract.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.extract'`

- [ ] **Step 3: Create `pipeline/extract.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_extract.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/adit/proxi-ai && git add pipeline/extract.py tests/pipeline/test_extract.py
git commit -m "feat: add Stage 2 signal extraction runner with resume logic"
```

---

## Task 7: Stage 3 — Auto-Optimized Clustering

**Files:**
- Create: `pipeline/cluster.py`
- Create: `tests/pipeline/test_cluster.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_cluster.py
import numpy as np
import pytest


def test_signals_to_text_combines_fields():
    from pipeline.cluster import _signals_to_text
    signals = {
        "pain_points": [{"signal": "breaks fast", "frequency": 3, "emotional_intensity": 0.8}],
        "desired_outcomes": [{"outcome": "lasts a year", "priority": 0.9}],
        "deal_breakers": ["no warranty"],
        "decision_factors_ranked": ["durability"],
    }
    text = _signals_to_text(signals)
    assert "breaks fast" in text
    assert "lasts a year" in text
    assert "no warranty" in text
    assert "durability" in text


def test_signals_to_text_empty_signals():
    from pipeline.cluster import _signals_to_text
    assert _signals_to_text({}) == ""


def test_score_clustering_returns_three_metrics():
    from pipeline.cluster import score_clustering
    rng = np.random.default_rng(42)
    embeddings = rng.random((20, 8))
    labels = np.array([0] * 10 + [1] * 10)
    metrics = score_clustering(embeddings, labels)
    assert set(metrics.keys()) == {"silhouette", "intra_similarity", "inter_distance"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_combined_score_uses_weights():
    from pipeline.cluster import combined_score
    metrics = {"silhouette": 1.0, "intra_similarity": 1.0, "inter_distance": 1.0}
    weights = {"silhouette": 0.5, "intra_similarity": 0.3, "inter_distance": 0.2}
    assert combined_score(metrics, weights) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_cluster.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.cluster'`

- [ ] **Step 3: Create `pipeline/cluster.py`**

```python
"""Stage 3: Embed signals and auto-optimize KMeans clustering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

SIGNALS_PATH = Path("data/extracted/signals.jsonl")
CLUSTERS_DIR = Path("data/clusters")
EMBEDDINGS_PATH = CLUSTERS_DIR / "embeddings.npy"
SWEEP_PATH = CLUSTERS_DIR / "sweep.json"
CLUSTERS_PATH = CLUSTERS_DIR / "clusters.json"


def _signals_to_text(signals: dict) -> str:
    """Flatten a signal dict into a single text blob for embedding."""
    parts: list[str] = []
    for pp in signals.get("pain_points", []):
        parts.append(pp.get("signal", ""))
    for do in signals.get("desired_outcomes", []):
        parts.append(do.get("outcome", ""))
    for db in signals.get("deal_breakers", []):
        if isinstance(db, str):
            parts.append(db)
    for df in signals.get("decision_factors_ranked", []):
        parts.append(df)
    return " ".join(filter(None, parts))


def embed_signals(signals_list: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [_signals_to_text(s) for s in signals_list]
    return model.encode(texts, show_progress_bar=True)


def score_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute silhouette, intra-cluster cosine similarity, inter-cluster distance."""
    n_clusters = len(set(labels))

    sil = float(silhouette_score(embeddings, labels)) if n_clusters > 1 else 0.0

    intra_sims: list[float] = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_embs = embeddings[mask]
        if len(cluster_embs) > 1:
            sim_matrix = cosine_similarity(cluster_embs)
            n = len(cluster_embs)
            upper = sim_matrix[np.triu_indices(n, k=1)]
            intra_sims.append(float(upper.mean()))
    intra = float(np.mean(intra_sims)) if intra_sims else 0.0

    centroids = np.array([embeddings[labels == c].mean(axis=0) for c in range(n_clusters)])
    if len(centroids) > 1:
        centroid_sims = cosine_similarity(centroids)
        n = len(centroids)
        upper = centroid_sims[np.triu_indices(n, k=1)]
        inter = float(1.0 - upper.mean())
    else:
        inter = 0.0

    return {"silhouette": sil, "intra_similarity": intra, "inter_distance": inter}


def combined_score(metrics: dict, weights: dict) -> float:
    return (
        metrics["silhouette"] * weights["silhouette"]
        + metrics["intra_similarity"] * weights["intra_similarity"]
        + metrics["inter_distance"] * weights["inter_distance"]
    )


def run_cluster(config: dict, n_clusters_override: int | None = None, force: bool = False) -> dict:
    """Cluster signals with auto-optimized or fixed n_clusters."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    signals_list: list[dict] = []
    with open(SIGNALS_PATH) as f:
        for line in f:
            signals_list.append(json.loads(line))

    # Embed (reuse checkpoint unless forced)
    if EMBEDDINGS_PATH.exists() and not force:
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        embeddings = embed_signals(signals_list)
        np.save(EMBEDDINGS_PATH, embeddings)

    weights = config["clustering"]["scoring_weights"]

    if n_clusters_override:
        best_k = n_clusters_override
        sweep_results: list[dict] = []
    else:
        sweep_range = config["clustering"]["sweep_range"]
        sweep_results = []
        best_k = sweep_range[0]
        best_score = -1.0

        for k in range(sweep_range[0], min(sweep_range[1] + 1, len(signals_list))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            metrics = score_clustering(embeddings, labels)
            cs = combined_score(metrics, weights)
            sweep_results.append({"n_clusters": k, **metrics, "combined_score": cs})
            if cs > best_score:
                best_score = cs
                best_k = k

        with open(SWEEP_PATH, "w") as f:
            json.dump(sweep_results, f, indent=2)

    # Final clustering with chosen k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = km.fit_predict(embeddings)
    final_metrics = score_clustering(embeddings, final_labels)

    result: dict[str, Any] = {
        "chosen_n_clusters": best_k,
        "sweep_winner": {
            **final_metrics,
            "combined_score": combined_score(final_metrics, weights),
        },
    }

    for c in range(best_k):
        mask = final_labels == c
        indices = np.where(mask)[0].tolist()
        centroid = embeddings[mask].mean(axis=0)

        dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        rep_local_idx = np.argsort(dists)[:5].tolist()
        rep_global_idx = [indices[i] for i in rep_local_idx]
        rep_batches = [signals_list[i] for i in rep_global_idx]

        all_pain: list[str] = []
        all_outcomes: list[str] = []
        all_deal_breakers: list[str] = []
        friction_counts: dict[str, int] = {}

        for idx in indices:
            s = signals_list[idx]
            all_pain.extend(pp["signal"] for pp in s.get("pain_points", []))
            all_outcomes.extend(do["outcome"] for do in s.get("desired_outcomes", []))
            all_deal_breakers.extend(
                db for db in s.get("deal_breakers", []) if isinstance(db, str)
            )
            ft = s.get("friction_tolerance", "medium")
            friction_counts[ft] = friction_counts.get(ft, 0) + 1

        dominant_friction = max(friction_counts, key=friction_counts.get) if friction_counts else "medium"

        result[f"cluster_{c}"] = {
            "label_hint": None,
            "member_count": int(mask.sum()),
            "representative_batches": rep_batches,
            "aggregate_signals": {
                "top_pain_points": list(dict.fromkeys(all_pain))[:10],
                "top_desired_outcomes": list(dict.fromkeys(all_outcomes))[:10],
                "top_deal_breakers": list(dict.fromkeys(all_deal_breakers))[:10],
                "dominant_friction_tolerance": dominant_friction,
            },
        }

    with open(CLUSTERS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_cluster.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/adit/proxi-ai && git add pipeline/cluster.py tests/pipeline/test_cluster.py
git commit -m "feat: add Stage 3 auto-optimized embedding and clustering"
```

---

## Task 8: Stage 4 — Export (Trait Extraction + Persona Synthesis + Dual Output)

**Files:**
- Create: `pipeline/export.py`
- Create: `tests/pipeline/test_export.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_export.py
import json
import pytest


SAMPLE_PERSONA = {
    "id": "college_student",
    "label": "College Student",
    "segment": {
        "context": "A student buying campus gear",
        "age_range": "18-24",
        "price_sensitivity": "very_high",
        "tech_savviness": "medium",
    },
    "goals": [{"goal": "find durable products", "priority": 1}],
    "constraints": ["tight budget"],
    "decision_weights": {"price": 0.6, "quality": 0.4},
    "behavioral_rules": ["checks price within 30 seconds"],
    "emotional_profile": {"baseline_patience": 0.4, "trust_starting_point": 0.5, "frustration_decay": 0.2},
    "deal_breakers": ["product breaks within a week"],
    "voice_sample": "I just need something that lasts through the semester.",
    "comparison_products": ["Brand A vs Brand B"],
    "browsing_patterns": {
        "typical_session_length_minutes": 5,
        "pages_before_decision": 3,
        "tab_behavior": "opens multiple tabs",
        "device": "laptop",
        "time_of_day": "evening",
    },
}

SAMPLE_TRAITS = [
    {
        "label": "Durability Concern",
        "description": "Products break quickly after purchase.",
        "key_phrases": ["snapped after a week", "fell apart"],
        "tone": "frustrated",
        "cluster_id": "cluster_0",
        "frequency": 0.38,
    },
    {
        "label": "Budget Consciousness",
        "description": "Highly sensitive to price and value.",
        "key_phrases": ["too expensive", "not worth it"],
        "tone": "cautious",
        "cluster_id": "cluster_1",
        "frequency": 0.28,
    },
]


def test_generate_markdown_contains_trait_labels():
    from pipeline.export import generate_markdown
    md = generate_markdown(SAMPLE_PERSONA, SAMPLE_TRAITS)
    assert "Durability Concern" in md
    assert "Budget Consciousness" in md


def test_generate_markdown_contains_key_phrases():
    from pipeline.export import generate_markdown
    md = generate_markdown(SAMPLE_PERSONA, SAMPLE_TRAITS)
    assert "snapped after a week" in md
    assert "too expensive" in md


def test_generate_markdown_contains_all_sections():
    from pipeline.export import generate_markdown
    md = generate_markdown(SAMPLE_PERSONA, SAMPLE_TRAITS)
    for section in ["## Overview", "## Key Traits", "## Behavioral Patterns",
                    "## Decision Priorities", "## Deal Breakers", "## Voice Sample"]:
        assert section in md


def test_build_rag_index_includes_required_fields():
    from pipeline.export import build_rag_index
    clusters = {
        "cluster_0": {
            "member_count": 5,
            "representative_batches": [
                {
                    "pain_points": [{"signal": "strap broke", "frequency": 2, "emotional_intensity": 0.8}],
                    "desired_outcomes": [{"outcome": "lasts all year", "priority": 0.9}],
                }
            ],
        }
    }
    entries = build_rag_index(SAMPLE_TRAITS[:1], clusters)
    assert len(entries) > 0
    for entry in entries:
        assert "review_id" in entry
        assert "trait_label" in entry
        assert "text" in entry
        assert "tone" in entry


def test_build_rag_index_uses_trait_label():
    from pipeline.export import build_rag_index
    clusters = {
        "cluster_0": {
            "member_count": 3,
            "representative_batches": [
                {"pain_points": [{"signal": "strap broke", "frequency": 2, "emotional_intensity": 0.7}], "desired_outcomes": []}
            ],
        }
    }
    entries = build_rag_index(SAMPLE_TRAITS[:1], clusters)
    assert all(e["trait_label"] == "Durability Concern" for e in entries)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_export.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.export'`

- [ ] **Step 3: Create `pipeline/export.py`**

```python
"""Stage 4: Trait extraction, persona synthesis, and dual-format export."""
from __future__ import annotations

import json
from pathlib import Path

from engine.llm import get_client, MODEL

CLUSTERS_PATH = Path("data/clusters/clusters.json")

_TRAIT_SYSTEM = """\
You are a behavioral analyst. Given a cluster of behavioral signals from product reviews,
identify the dominant trait this cluster represents.

Return ONLY valid JSON (no markdown fences):
{
  "label": "<short trait name, e.g. 'Durability Concern'>",
  "description": "<1-2 sentences describing this trait>",
  "key_phrases": ["<verbatim phrase 1>", "<verbatim phrase 2>", "<verbatim phrase 3>"],
  "tone": "<one of: frustrated, cautious, aspirational, pragmatic, skeptical, enthusiastic>"
}"""

_PERSONA_SYSTEM = """\
You are an expert buyer-persona researcher. Given behavioral traits from product reviews for
college students, synthesize a complete, grounded buyer persona.

Return ONLY valid JSON (no markdown fences) matching this schema exactly:
{
  "id": "college_student",
  "label": "College Student",
  "segment": {
    "company_size": "N/A",
    "role": "Student",
    "tech_savviness": "<low|medium|high>",
    "price_sensitivity": "<low|medium|high|very_high>",
    "risk_tolerance": "<low|medium|high>",
    "age_range": "18-24",
    "context": "<1 sentence>"
  },
  "goals": [{"goal": "<string>", "priority": <int 1-5>}],
  "constraints": ["<string>"],
  "decision_weights": {"<factor>": <float, all sum to 1.0>},
  "behavioral_rules": ["<rule with action verb or number>"],
  "emotional_profile": {
    "baseline_patience": <float 0-1>,
    "trust_starting_point": <float 0-1>,
    "frustration_decay": <float 0-1>
  },
  "deal_breakers": ["<concrete, binary condition>"],
  "voice_sample": "<1-2 paragraph quote in authentic student voice>",
  "comparison_products": ["<product category comparisons>"],
  "browsing_patterns": {
    "typical_session_length_minutes": <int>,
    "pages_before_decision": <int>,
    "tab_behavior": "<string>",
    "device": "<string>",
    "time_of_day": "<string>"
  }
}

Rules: decision_weights must sum to 1.0; provide at least 4 goals, 3 constraints,
6 behavioral_rules, 3 deal_breakers."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_trait(cluster_id: str, cluster_data: dict, client) -> dict:
    """Ask DeepSeek to label a cluster as a behavioral trait."""
    signals_text = json.dumps(cluster_data["aggregate_signals"], indent=2)
    sample_phrases: list[str] = []
    for batch in cluster_data.get("representative_batches", [])[:2]:
        sample_phrases.extend(pp["signal"] for pp in batch.get("pain_points", [])[:3])
        sample_phrases.extend(do["outcome"] for do in batch.get("desired_outcomes", [])[:2])

    user_content = (
        f"Cluster ID: {cluster_id}\n\n"
        f"Aggregate signals:\n{signals_text}\n\n"
        f"Sample phrases:\n" + "\n".join(f"- {p}" for p in sample_phrases[:10])
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _TRAIT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    trait = json.loads(_strip_fences(response.choices[0].message.content))
    trait["cluster_id"] = cluster_id
    return trait


def synthesize_persona(traits: list[dict], clusters: dict, client) -> dict:
    """Synthesize full persona from all cluster traits."""
    total = sum(
        clusters[k]["member_count"] for k in clusters if k.startswith("cluster_")
    )
    for trait in traits:
        cid = trait["cluster_id"]
        trait["frequency"] = round(clusters[cid]["member_count"] / total, 3) if total else 0.0

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": _PERSONA_SYSTEM},
            {"role": "user", "content": f"Traits from {total} batches:\n\n{json.dumps(traits, indent=2)}"},
        ],
    )
    return json.loads(_strip_fences(response.choices[0].message.content))


def build_rag_index(traits: list[dict], clusters: dict) -> list[dict]:
    """Build RAG index entries from representative review snippets."""
    entries: list[dict] = []
    for trait in traits:
        cid = trait["cluster_id"]
        cluster_data = clusters.get(cid, {})
        for b_idx, batch in enumerate(cluster_data.get("representative_batches", [])):
            for pp_idx, pp in enumerate(batch.get("pain_points", [])[:3]):
                entries.append({
                    "review_id": f"{cid}_b{b_idx}_pp{pp_idx}",
                    "trait_label": trait["label"],
                    "text": pp["signal"],
                    "tone": trait.get("tone", "unknown"),
                })
            for do_idx, do in enumerate(batch.get("desired_outcomes", [])[:2]):
                entries.append({
                    "review_id": f"{cid}_b{b_idx}_do{do_idx}",
                    "trait_label": trait["label"],
                    "text": do["outcome"],
                    "tone": trait.get("tone", "unknown"),
                })
    return entries


def generate_markdown(persona: dict, traits: list[dict]) -> str:
    """Generate a narrative Markdown context doc from the persona JSON."""
    seg = persona.get("segment", {})
    lines = [
        f"# {persona.get('label', 'Persona')} — Persona Context Document",
        "",
        "## Overview",
        "",
        seg.get("context", ""),
        "",
        f"- **Age range:** {seg.get('age_range', 'N/A')}",
        f"- **Price sensitivity:** {seg.get('price_sensitivity', 'N/A')}",
        f"- **Tech savviness:** {seg.get('tech_savviness', 'N/A')}",
        "",
        "## Key Traits",
        "",
    ]

    for trait in sorted(traits, key=lambda t: -t.get("frequency", 0)):
        freq_pct = f"{trait.get('frequency', 0):.0%}"
        lines += [
            f"### {trait['label']} (frequency: {freq_pct})",
            "",
            trait.get("description", ""),
            "",
            f"**Tone:** {trait.get('tone', 'N/A')}",
            "",
            "**Sample phrases:**",
        ]
        for phrase in trait.get("key_phrases", []):
            lines.append(f'- "{phrase}"')
        lines.append("")

    lines += ["## Behavioral Patterns", ""]
    for rule in persona.get("behavioral_rules", []):
        lines.append(f"- {rule}")
    lines.append("")

    lines += ["## Decision Priorities", ""]
    weights = persona.get("decision_weights", {})
    for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
        lines.append(f"- **{factor}:** {weight:.0%}")
    lines.append("")

    lines += ["## Deal Breakers", ""]
    for db in persona.get("deal_breakers", []):
        lines.append(f"- {db}")
    lines.append("")

    lines += ["## Voice Sample", "", f'> {persona.get("voice_sample", "")}', ""]

    return "\n".join(lines)


def run_export(persona_id: str = "college_student", force: bool = False) -> None:
    """Run Stage 4: extract traits, synthesize persona, write all outputs."""
    persona_dir = Path(f"data/personas/{persona_id}")
    json_path = persona_dir / "persona.json"

    if json_path.exists() and not force:
        print(f"[export] {json_path} already exists. Use --force to regenerate.")
        return

    persona_dir.mkdir(parents=True, exist_ok=True)

    with open(CLUSTERS_PATH) as f:
        clusters = json.load(f)

    client = get_client()
    cluster_keys = [k for k in clusters if k.startswith("cluster_")]
    traits: list[dict] = []

    for cid in cluster_keys:
        try:
            trait = extract_trait(cid, clusters[cid], client)
        except Exception as e:
            print(f"[export] Warning: trait extraction failed for {cid}: {e}")
            trait = {
                "label": cid,
                "description": "Trait extraction failed — using cluster ID as fallback label.",
                "key_phrases": [],
                "tone": "unknown",
                "cluster_id": cid,
            }
        traits.append(trait)

    persona = synthesize_persona(traits, clusters, client)
    persona["traits"] = [
        {
            "label": t["label"],
            "description": t["description"],
            "key_phrases": t["key_phrases"],
            "frequency": t.get("frequency", 0.0),
            "tone": t["tone"],
            "representative_review_ids": [
                f"{t['cluster_id']}_b0_pp0",
                f"{t['cluster_id']}_b0_pp1",
            ],
        }
        for t in traits
    ]

    with open(json_path, "w") as f:
        json.dump(persona, f, indent=2)

    md_path = persona_dir / "persona.md"
    with open(md_path, "w") as f:
        f.write(generate_markdown(persona, traits))

    rag_path = persona_dir / "rag_index.jsonl"
    with open(rag_path, "w") as f:
        for entry in build_rag_index(traits, clusters):
            f.write(json.dumps(entry) + "\n")

    print(f"[export] Written:\n  {json_path}\n  {md_path}\n  {rag_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/pipeline/test_export.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/adit/proxi-ai && git add pipeline/export.py tests/pipeline/test_export.py
git commit -m "feat: add Stage 4 export with trait extraction, persona synthesis, JSON+Markdown+RAG"
```

---

## Task 9: Pipeline CLI

**Files:**
- Create: `pipeline.py`

- [ ] **Step 1: Create `pipeline.py`**

```python
#!/usr/bin/env python3
"""Proxi AI — Persona Training Pipeline CLI

Commands:
    collect     Stage 1: collect reviews (HuggingFace + Playwright)
    extract     Stage 2: extract behavioral signals
    cluster     Stage 3: embed and cluster signals
    export      Stage 4: synthesize persona, write JSON + Markdown
    run-all     Run all 4 stages in sequence
"""
from __future__ import annotations

import asyncio

import click
import yaml
from rich.console import Console

console = Console()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """Proxi AI — Persona Training Pipeline"""


@cli.command()
@click.option("--force", is_flag=True, help="Re-collect even if reviews.jsonl exists")
@click.option("--hf-only", is_flag=True, help="HuggingFace only, skip Playwright")
@click.option("--scrape-only", is_flag=True, help="Playwright only, skip HuggingFace")
def collect(force: bool, hf_only: bool, scrape_only: bool) -> None:
    """Stage 1: Collect reviews from HuggingFace + Playwright."""
    from pathlib import Path
    from pipeline.collect import collect_huggingface, load_existing_hashes, scrape_amazon

    reviews_path = Path("data/raw/reviews.jsonl")
    if reviews_path.exists() and not force:
        console.print("[yellow]reviews.jsonl already exists. Use --force to re-collect.[/yellow]")
        return

    config = _load_config()
    existing_hashes = load_existing_hashes()

    if not scrape_only:
        console.print("[bold]Pulling from HuggingFace...[/bold]")
        n = collect_huggingface(config, existing_hashes)
        console.print(f"[green]HuggingFace: {n} reviews collected[/green]")

    if not hf_only:
        console.print("[bold]Running Playwright scraper...[/bold]")
        n = asyncio.run(scrape_amazon(config, existing_hashes))
        console.print(f"[green]Playwright: {n} reviews scraped[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Re-extract even if signals.jsonl exists")
@click.option("--batch-size", default=30, show_default=True, help="Reviews per extraction batch")
def extract(force: bool, batch_size: int) -> None:
    """Stage 2: Extract behavioral signals from reviews."""
    from pipeline.extract import run_extract
    console.print("[bold]Extracting signals...[/bold]")
    n = run_extract(batch_size=batch_size, force=force)
    console.print(f"[green]Processed {n} new batches[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Re-cluster even if clusters.json exists")
@click.option("--n-clusters", default=None, type=int, help="Override auto-optimization")
def cluster(force: bool, n_clusters: int | None) -> None:
    """Stage 3: Embed signals and auto-optimize clusters."""
    from pipeline.cluster import run_cluster
    config = _load_config()
    console.print("[bold]Clustering signals...[/bold]")
    result = run_cluster(config, n_clusters_override=n_clusters, force=force)
    chosen = result["chosen_n_clusters"]
    console.print(f"[green]Clustered into {chosen} groups[/green]")


@cli.command()
@click.option("--force", is_flag=True, help="Re-export even if persona.json exists")
@click.option("--persona-id", default="college_student", show_default=True, help="Output persona ID")
def export(force: bool, persona_id: str) -> None:
    """Stage 4: Synthesize persona and export JSON + Markdown."""
    from pipeline.export import run_export
    console.print("[bold]Synthesizing persona...[/bold]")
    run_export(persona_id=persona_id, force=force)
    console.print(f"[green]Persona exported to data/personas/{persona_id}/[/green]")


@cli.command("run-all")
@click.option("--force", is_flag=True, help="Force re-run all stages")
@click.option("--persona-id", default="college_student", show_default=True)
@click.pass_context
def run_all(ctx: click.Context, force: bool, persona_id: str) -> None:
    """Run all 4 pipeline stages in sequence."""
    ctx.invoke(collect, force=force, hf_only=False, scrape_only=False)
    ctx.invoke(extract, force=force, batch_size=30)
    ctx.invoke(cluster, force=force, n_clusters=None)
    ctx.invoke(export, force=force, persona_id=persona_id)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Verify CLI loads**

```bash
cd /Users/adit/proxi-ai && python pipeline.py --help
```
Expected output:
```
Usage: pipeline.py [OPTIONS] COMMAND [ARGS]...

  Proxi AI — Persona Training Pipeline

Options:
  --help  Show this message and exit.

Commands:
  collect  Stage 1: Collect reviews from HuggingFace + Playwright.
  extract  Stage 2: Extract behavioral signals from reviews.
  cluster  Stage 3: Embed signals and auto-optimize clusters.
  export   Stage 4: Synthesize persona and export JSON + Markdown.
  run-all  Run all 4 pipeline stages in sequence.
```

- [ ] **Step 3: Verify each subcommand help loads**

```bash
cd /Users/adit/proxi-ai && python pipeline.py collect --help && python pipeline.py extract --help && python pipeline.py cluster --help && python pipeline.py export --help
```
Expected: each shows options without errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/adit/proxi-ai && git add pipeline.py
git commit -m "feat: add pipeline.py CLI with collect/extract/cluster/export/run-all commands"
```

---

## Task 10: Full Test Suite + .env Setup

**Files:**
- Modify: `.env` (local only, not committed)

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/adit/proxi-ai && python -m pytest tests/ -v
```
Expected: all tests pass (12 tests across 5 test files).

- [ ] **Step 2: Set up .env with OpenRouter key**

```bash
cd /Users/adit/proxi-ai && echo "OPENROUTER_API_KEY=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" > .env
```

- [ ] **Step 3: Verify OpenRouter connection**

```bash
cd /Users/adit/proxi-ai && python -c "
from dotenv import load_dotenv; load_dotenv()
from engine.llm import get_client, MODEL
client = get_client()
resp = client.chat.completions.create(
    model=MODEL,
    max_tokens=20,
    messages=[{'role': 'user', 'content': 'Say hello in 5 words.'}]
)
print(resp.choices[0].message.content)
"
```
Expected: a short greeting phrase printed without error.

- [ ] **Step 4: Commit all remaining files**

```bash
cd /Users/adit/proxi-ai && git add requirements.txt
git commit -m "chore: finalize requirements.txt with openai dependency"
```

---

## Task 11: Run the Pipeline End-to-End

This task actually executes the pipeline to produce the context doc. Run stages individually so you can monitor and resume.

- [ ] **Step 1: Collect reviews (HuggingFace first, scraper second)**

```bash
cd /Users/adit/proxi-ai && python -c "from dotenv import load_dotenv; load_dotenv()" && \
  python pipeline.py collect --hf-only
```
Expected: progress output per category, `data/raw/reviews.jsonl` created. Check size:
```bash
wc -l data/raw/reviews.jsonl
```
Expected: at least 10,000 lines.

- [ ] **Step 2: Run Playwright gap-fill**

```bash
cd /Users/adit/proxi-ai && python pipeline.py collect --scrape-only
```
Expected: scraper logs progress per query, appends to `reviews.jsonl`. Check for errors:
```bash
cat data/raw/scrape_errors.jsonl 2>/dev/null || echo "No scrape errors"
```

- [ ] **Step 3: Extract signals (resumable — safe to re-run if interrupted)**

```bash
cd /Users/adit/proxi-ai && python pipeline.py extract
```
Expected: batch progress logged, `data/extracted/signals.jsonl` grows. To monitor progress:
```bash
wc -l data/extracted/signals.jsonl
```

- [ ] **Step 4: Cluster signals**

```bash
cd /Users/adit/proxi-ai && python pipeline.py cluster
```
Expected: output like `Clustered into N groups`. Check sweep results:
```bash
python -c "import json; d=json.load(open('data/clusters/sweep.json')); [print(r) for r in d]"
```

- [ ] **Step 5: Export persona**

```bash
cd /Users/adit/proxi-ai && python pipeline.py export
```
Expected:
```
[export] Written:
  data/personas/college_student/persona.json
  data/personas/college_student/persona.md
  data/personas/college_student/rag_index.jsonl
```

- [ ] **Step 6: Verify persona.json is valid and complete**

```bash
cd /Users/adit/proxi-ai && python -c "
import json
p = json.load(open('data/personas/college_student/persona.json'))
print('Label:', p['label'])
print('Traits:', len(p.get('traits', [])))
print('Behavioral rules:', len(p.get('behavioral_rules', [])))
print('Deal breakers:', len(p.get('deal_breakers', [])))
print('Decision weights sum:', round(sum(p.get('decision_weights', {}).values()), 3))
"
```
Expected: label prints as `College Student`, all counts > 0, decision weights sum to `1.0`.

- [ ] **Step 7: Verify persona.md is readable**

```bash
cd /Users/adit/proxi-ai && head -60 data/personas/college_student/persona.md
```
Expected: readable Markdown with Overview, Key Traits sections visible.

- [ ] **Step 8: Commit generated outputs**

```bash
cd /Users/adit/proxi-ai && git add data/personas/college_student/persona.json data/personas/college_student/persona.md data/personas/college_student/rag_index.jsonl
git commit -m "feat: add generated college_student persona spec (JSON + Markdown + RAG index)"
```
