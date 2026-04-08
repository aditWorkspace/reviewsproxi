"""Hacker News comment scraper via Algolia Search API.

No API key required.  Algolia's HN mirror is publicly accessible and
handles ~10k requests/hour comfortably.

API docs: https://hn.algolia.com/api
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests

ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search"
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ProxiAI/1.0 (persona training; contact: proxi-ai)"})


def _clean(html: str) -> str:
    text = _HTML_TAG.sub(" ", html or "")
    return _WHITESPACE.sub(" ", text).strip()


def _search_page(query: str, page: int, hits_per_page: int = 200) -> dict:
    resp = _SESSION.get(
        ALGOLIA_SEARCH,
        params={
            "query": query,
            "tags": "comment",
            "hitsPerPage": hits_per_page,
            "page": page,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def search_hn_comments(
    query: str,
    max_results: int = 300,
    min_points: int = 1,
    log_fn=None,
) -> list[dict[str, Any]]:
    """Return up to *max_results* HN comments matching *query*.

    Each result dict contains:
      text, source, source_type, upvotes, story_title, story_url,
      author, created_at, hn_id
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    page = 0

    while len(results) < max_results:
        try:
            data = _search_page(query, page, hits_per_page=min(200, max_results))
        except requests.RequestException as exc:
            if log_fn:
                log_fn("warn", f"HN API error on page {page}: {exc}")
            break

        hits = data.get("hits", [])
        if not hits:
            break

        for hit in hits:
            oid = hit.get("objectID", "")
            if oid in seen:
                continue
            seen.add(oid)

            text = _clean(hit.get("comment_text") or "")
            if len(text) < 60:
                continue

            points = hit.get("points") or 0
            if points < min_points:
                continue

            results.append({
                "text": text,
                "source": "hackernews",
                "source_type": "hackernews",
                "upvotes": points,
                "story_title": hit.get("story_title") or "",
                "story_url": hit.get("story_url") or "",
                "author": hit.get("author") or "",
                "created_at": hit.get("created_at") or "",
                "hn_id": oid,
            })

            if len(results) >= max_results:
                break

        nb_pages = data.get("nbPages", 1)
        page += 1
        if page >= nb_pages:
            break

        time.sleep(0.15)   # ~6–7 req/s — well within Algolia's free tier

    return results


def scrape_hn_for_project(
    queries: list[str],
    max_per_query: int = 250,
    min_upvotes: int = 1,
    log_fn=None,
) -> list[dict[str, Any]]:
    """Scrape HN comments for a list of search queries.

    Parameters
    ----------
    queries:
        Search terms derived from persona products + demographic keywords.
        Typically 4-8 targeted queries.
    max_per_query:
        Hard cap per query to prevent one topic from dominating.
    min_upvotes:
        Drop comments with fewer upvotes than this (filters noise).
    log_fn:
        Optional callable(kind, msg) for progress logging.

    Returns
    -------
    list[dict]
        Deduplicated comment dicts ready for append_reviews().
    """
    seen_ids: set[str] = set()
    all_comments: list[dict[str, Any]] = []

    for query in queries:
        if log_fn:
            log_fn("info", f"  HN search: '{query}'…")

        comments = search_hn_comments(
            query,
            max_results=max_per_query,
            min_points=min_upvotes,
            log_fn=log_fn,
        )

        added = 0
        for c in comments:
            if c["hn_id"] in seen_ids:
                continue
            seen_ids.add(c["hn_id"])
            all_comments.append(c)
            added += 1

        if log_fn:
            log_fn("ok", f"    +{added} unique comments (total so far: {len(all_comments)})")

    return all_comments
