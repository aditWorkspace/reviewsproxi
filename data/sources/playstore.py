"""Google Play Store review scraper.

Uses google-play-scraper to search for apps and pull reviews.
No API key required.
"""
from __future__ import annotations

import time
from typing import Any


def scrape_playstore(
    queries: list[str],
    max_per_app: int = 300,
    log_fn=None,
) -> list[dict[str, Any]]:
    """Search Play Store for each query and pull reviews.

    Parameters
    ----------
    queries:      Product / app name search terms (reuse intel["products"]).
    max_per_app:  Max reviews per matched app.
    log_fn:       Optional callable(kind, msg).

    Returns
    -------
    list[dict]   Review dicts with source_type="playstore".
    """
    try:
        from google_play_scraper import reviews as gp_reviews, search as gp_search, Sort
    except ImportError:
        if log_fn:
            log_fn("warn", "google-play-scraper not installed — skipping Play Store")
        return []

    seen_pkg: set[str] = set()
    all_reviews: list[dict[str, Any]] = []

    for query in queries:
        # Search for the app
        try:
            results = gp_search(query, lang="en", country="us", n_hits=3)
        except Exception as exc:
            if log_fn:
                log_fn("warn", f"  Play Store search failed for '{query}': {exc}")
            continue

        if not results:
            if log_fn:
                log_fn("warn", f"  Play Store: no app found for '{query}'")
            continue

        pkg = results[0].get("appId", "")
        title = results[0].get("title", query)

        if not pkg or pkg in seen_pkg:
            continue
        seen_pkg.add(pkg)

        if log_fn:
            log_fn("info", f"  Play Store: '{title}' ({pkg})…")

        try:
            raw, _ = gp_reviews(
                pkg,
                lang="en",
                country="us",
                sort=Sort.MOST_RELEVANT,
                count=max_per_app,
            )
        except Exception as exc:
            if log_fn:
                log_fn("warn", f"  Play Store scrape failed for '{title}': {exc}")
            continue

        added = 0
        for r in raw:
            text = (r.get("content") or "").strip()
            if len(text) < 40:
                continue
            rating = r.get("score")
            all_reviews.append({
                "text": text,
                "source": "playstore",
                "source_type": "playstore",
                "rating": float(rating) if rating else None,
                "title": "",
                "author": r.get("userName", ""),
                "created_at": str(r.get("at", "")),
                "app_name": title,
                "app_id": pkg,
                "upvotes": r.get("thumbsUpCount", 0),
            })
            added += 1

        if log_fn:
            log_fn("ok", f"    +{added} Play Store reviews for '{title}'")
        time.sleep(0.5)

    return all_reviews
