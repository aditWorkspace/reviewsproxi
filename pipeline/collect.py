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
