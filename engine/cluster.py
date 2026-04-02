"""Persona clustering module.

Groups reviewers into behavioral clusters based on review patterns,
producing representative profiles for downstream persona synthesis.
"""

import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Keyword lists for signal detection
# ---------------------------------------------------------------------------

PRICE_KEYWORDS: list[str] = [
    "price", "pricing", "cost", "expensive", "cheap", "affordable",
    "budget", "value", "overpriced", "underpriced", "discount", "deal",
    "fee", "fees", "subscription", "plan", "tier", "worth",
]

QUALITY_KEYWORDS: list[str] = [
    "quality", "reliable", "unreliable", "buggy", "polished", "stable",
    "crash", "crashes", "downtime", "performance", "fast", "slow",
    "laggy", "smooth", "robust", "flimsy", "solid", "broken",
]

COMPARISON_KEYWORDS: list[str] = [
    "compared to", "versus", "vs", "better than", "worse than",
    "alternative", "alternatives", "switched from", "moved from",
    "competitor", "competitors", "unlike", "similar to",
]

# Simple positive / negative word lists for emotional valence scoring.
_POSITIVE_WORDS: set[str] = {
    "love", "great", "excellent", "amazing", "awesome", "fantastic",
    "wonderful", "best", "happy", "pleased", "impressed", "recommend",
    "intuitive", "easy", "seamless", "delight", "perfect", "superb",
}

_NEGATIVE_WORDS: set[str] = {
    "hate", "terrible", "awful", "worst", "horrible", "frustrating",
    "disappointed", "annoying", "useless", "broken", "poor", "bad",
    "confusing", "clunky", "ugly", "painful", "nightmare", "regret",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Return the number of keyword occurrences found in *text*."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        # Use word-boundary search so "vs" doesn't match inside "canvas".
        count += len(re.findall(rf"\b{re.escape(kw)}\b", text_lower))
    return count


def _emotional_valence(text: str) -> float:
    """Return a valence score in [-1.0, 1.0].

    Computed as (positive_hits - negative_hits) / total_hits, or 0.0 when
    there are no sentiment words at all.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reviewer_profiles(
    reviews_by_reviewer: dict[str, list[dict]],
) -> list[dict]:
    """Create a behavioural-feature vector for every reviewer.

    Parameters
    ----------
    reviews_by_reviewer:
        Mapping of ``reviewer_id`` to a list of review dicts.  Each review
        dict is expected to have at least ``"text"`` (str), ``"rating"``
        (numeric), and optionally ``"category"`` (str).

    Returns
    -------
    list[dict]
        One dict per reviewer with the following keys:
        ``reviewer_id``, ``avg_rating``, ``price_mentions``,
        ``quality_mentions``, ``comparison_mentions``,
        ``review_length_avg``, ``emotional_valence``,
        ``categories_reviewed``, ``review_count``, ``reviews``.
    """
    profiles: list[dict] = []

    for reviewer_id, reviews in reviews_by_reviewer.items():
        if not reviews:
            continue

        ratings: list[float] = []
        price_hits = 0
        quality_hits = 0
        comparison_hits = 0
        lengths: list[int] = []
        valences: list[float] = []
        categories: set[str] = set()

        for review in reviews:
            text: str = review.get("text", "")
            rating = review.get("rating")
            category = review.get("category")

            if rating is not None:
                ratings.append(float(rating))

            price_hits += _count_keyword_hits(text, PRICE_KEYWORDS)
            quality_hits += _count_keyword_hits(text, QUALITY_KEYWORDS)
            comparison_hits += _count_keyword_hits(text, COMPARISON_KEYWORDS)
            lengths.append(len(text))
            valences.append(_emotional_valence(text))

            if category:
                categories.add(category)

        n = len(reviews)
        profiles.append({
            "reviewer_id": reviewer_id,
            "avg_rating": np.mean(ratings).item() if ratings else 0.0,
            "price_mentions": price_hits / n,
            "quality_mentions": quality_hits / n,
            "comparison_mentions": comparison_hits / n,
            "review_length_avg": np.mean(lengths).item() if lengths else 0.0,
            "emotional_valence": np.mean(valences).item() if valences else 0.0,
            "categories_reviewed": len(categories),
            "review_count": n,
            "reviews": reviews,
        })

    return profiles


def cluster_reviewers(
    profiles: list[dict],
    n_clusters: int = 5,
) -> dict:
    """Cluster reviewer profiles using KMeans.

    Parameters
    ----------
    profiles:
        Output of :func:`build_reviewer_profiles`.
    n_clusters:
        Number of clusters to produce.

    Returns
    -------
    dict
        ``"labels"`` – list of cluster labels aligned with *profiles*.
        ``"clusters"`` – mapping of cluster label (int) to a dict with
        ``"profiles"`` (the profiles in that cluster) and
        ``"representative_reviews"`` (a flat list of reviews from the
        cluster member closest to the centroid).
    """
    if not profiles:
        return {"labels": [], "clusters": {}}

    # Clamp n_clusters to the number of available profiles.
    n_clusters = min(n_clusters, len(profiles))

    feature_keys = [
        "avg_rating",
        "price_mentions",
        "quality_mentions",
        "comparison_mentions",
        "review_length_avg",
        "emotional_valence",
        "categories_reviewed",
    ]

    X = np.array([[p[k] for k in feature_keys] for p in profiles], dtype=np.float64)

    # Normalise features to zero-mean / unit-variance for KMeans.
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    X_norm = (X - means) / stds

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_norm).tolist()

    # Organise profiles by cluster and pick representatives.
    clusters: dict[int, dict] = defaultdict(lambda: {
        "profiles": [],
        "representative_reviews": [],
    })

    for idx, label in enumerate(labels):
        clusters[label]["profiles"].append(profiles[idx])

    # For each cluster, select the profile closest to the centroid.
    for label in range(n_clusters):
        member_indices = [i for i, l in enumerate(labels) if l == label]
        if not member_indices:
            continue

        centroid = kmeans.cluster_centers_[label]
        member_vectors = X_norm[member_indices]
        distances = np.linalg.norm(member_vectors - centroid, axis=1)
        closest_local = int(np.argmin(distances))
        closest_global = member_indices[closest_local]

        clusters[label]["representative_reviews"] = profiles[closest_global]["reviews"]

    return {
        "labels": labels,
        "clusters": dict(clusters),
    }
