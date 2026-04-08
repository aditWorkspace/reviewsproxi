"""Apple App Store review scraper.

Uses the iTunes Search API to find app IDs, then app-store-scraper to pull
reviews. No API key required.
"""
from __future__ import annotations

import time
from typing import Any

import requests

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ProxiAI/1.0"})

ITUNES_SEARCH = "https://itunes.apple.com/search"


def _find_app_id(query: str) -> tuple[str, str] | None:
    """Return (app_id, app_name) for the top App Store result, or None."""
    try:
        r = _SESSION.get(
            ITUNES_SEARCH,
            params={"term": query, "entity": "software", "limit": 3, "country": "us"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return str(results[0]["trackId"]), results[0]["trackName"]
    except Exception:
        pass
    return None


def scrape_appstore(
    queries: list[str],
    max_per_app: int = 300,
    log_fn=None,
) -> list[dict[str, Any]]:
    """Search App Store for each query and pull reviews.

    Parameters
    ----------
    queries:      Product / app name search terms (reuse intel["products"]).
    max_per_app:  Max reviews per matched app.
    log_fn:       Optional callable(kind, msg).

    Returns
    -------
    list[dict]   Review dicts with source_type="appstore".
    """
    try:
        from app_store_scraper import AppStore
    except ImportError:
        if log_fn:
            log_fn("warn", "app-store-scraper not installed — skipping App Store")
        return []

    seen_ids: set[str] = set()
    all_reviews: list[dict[str, Any]] = []

    for query in queries:
        hit = _find_app_id(query)
        if not hit:
            if log_fn:
                log_fn("warn", f"  App Store: no app found for '{query}'")
            continue

        app_id, app_name = hit
        if app_id in seen_ids:
            continue
        seen_ids.add(app_id)

        if log_fn:
            log_fn("info", f"  App Store: '{app_name}' (id={app_id})…")

        try:
            scraper = AppStore(country="us", app_name=app_name, app_id=app_id)
            scraper.review(how_many=max_per_app)
            raw = scraper.reviews or []
        except Exception as exc:
            if log_fn:
                log_fn("warn", f"  App Store scrape failed for '{app_name}': {exc}")
            continue

        added = 0
        for r in raw:
            text = (r.get("review") or "").strip()
            if len(text) < 40:
                continue
            rating = r.get("rating")
            all_reviews.append({
                "text": text,
                "source": "appstore",
                "source_type": "appstore",
                "rating": float(rating) if rating else None,
                "title": r.get("title", ""),
                "author": r.get("userName", ""),
                "created_at": str(r.get("date", "")),
                "app_name": app_name,
                "app_id": app_id,
            })
            added += 1

        if log_fn:
            log_fn("ok", f"    +{added} App Store reviews for '{app_name}'")
        time.sleep(0.5)

    return all_reviews
