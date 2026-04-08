"""Signal aggregation and deduplication module.

Merges extraction results across batches using embedding-based similarity
clustering (sentence-transformers) so that near-duplicate signals are
collapsed and ranked by frequency and emotional intensity.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level model (lazy-loaded)
# ---------------------------------------------------------------------------

_EMBED_MODEL: SentenceTransformer | None = None
_EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"


def _get_embed_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer, loading it on first call."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


# ---------------------------------------------------------------------------
# Similarity clustering
# ---------------------------------------------------------------------------


def cluster_by_similarity(
    signals: list[dict],
    text_key: str = "signal",
    threshold: float = 0.85,
) -> list[dict]:
    """Cluster a flat list of signal dicts by semantic similarity.

    Each cluster is represented by the member whose embedding is closest to
    the cluster centroid (not arbitrarily the first member).  Frequency is
    set to the cluster size — a real observation count derived from how many
    independent batches produced a semantically similar signal.

    Parameters
    ----------
    signals:
        List of signal dicts.  Each must contain a string field named
        *text_key* (default ``"signal"``).
    text_key:
        The dict key whose value is used for embedding comparison.
    threshold:
        Cosine-similarity threshold above which two signals are merged.

    Returns
    -------
    list[dict]
        Deduplicated signals with updated counts.
    """
    if not signals:
        return []

    texts: list[str] = [s.get(text_key, "") for s in signals]
    model = _get_embed_model()
    embeddings: np.ndarray = model.encode(texts, normalize_embeddings=True)

    # Greedy leader clustering — note this is still order-dependent; a future
    # improvement is to replace with complete-linkage or HDBSCAN.
    n = len(signals)
    assigned = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Merge each cluster into one canonical signal.
    merged: list[dict] = []
    for cluster_indices in clusters:
        members = [signals[idx] for idx in cluster_indices]

        # Select the member whose embedding is closest to the cluster centroid,
        # rather than blindly picking the first (insertion-order) member.
        cluster_embs = embeddings[np.array(cluster_indices)]
        centroid = cluster_embs.mean(axis=0)
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        best_local = int(np.argmin(distances))
        canonical = dict(members[best_local])

        # Frequency = cluster size: a deterministic count of how many batches
        # independently produced a semantically similar signal.  This replaces
        # the previous approach of summing LLM-fabricated per-batch counts.
        canonical["frequency"] = len(members)

        # Average emotional intensity across all cluster members; only use
        # members that actually have an intensity value (don't default to 0.5).
        intensities = [
            m["emotional_intensity"]
            for m in members
            if m.get("emotional_intensity") is not None
        ]
        canonical["emotional_intensity"] = round(sum(intensities) / len(intensities), 3) if intensities else None
        canonical["_cluster_size"] = len(members)

        merged.append(canonical)

    return merged


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _merge_keyed_signals(
    all_entries: list[dict],
    text_key: str,
    threshold: float = 0.85,
    top_n: int = 15,
) -> list[dict]:
    """Cluster, sort by score, and return top *top_n* entries."""
    clustered = cluster_by_similarity(all_entries, text_key=text_key, threshold=threshold)

    # Score = frequency (real cluster-size count) × emotional_intensity.
    # When intensity is unknown we use 0.5 as a neutral prior rather than
    # promoting or demoting the signal artificially.
    for entry in clustered:
        intensity = entry.get("emotional_intensity")
        entry["_score"] = entry["frequency"] * (intensity if intensity is not None else 0.5)

    clustered.sort(key=lambda x: x["_score"], reverse=True)
    return clustered[:top_n]


def _merge_string_lists(all_items: list[str], top_n: int = 15) -> list[str]:
    """Deduplicate a list of strings using embeddings, return top by count."""
    if not all_items:
        return []

    model = _get_embed_model()
    unique_map: dict[str, int] = {}
    for item in all_items:
        item_clean = item.strip()
        if not item_clean:
            continue
        unique_map[item_clean] = unique_map.get(item_clean, 0) + 1

    if not unique_map:
        return []

    labels = list(unique_map.keys())
    counts = [unique_map[lb] for lb in labels]
    embeddings = model.encode(labels, normalize_embeddings=True)

    n = len(labels)
    assigned = [False] * n
    result: list[tuple[str, int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster_indices = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if float(np.dot(embeddings[i], embeddings[j])) >= 0.85:
                cluster_indices.append(j)
                assigned[j] = True

        total_count = sum(counts[idx] for idx in cluster_indices)

        # Pick the label closest to the cluster centroid, not the first one.
        if len(cluster_indices) == 1:
            best_label = labels[cluster_indices[0]]
        else:
            cluster_embs = embeddings[np.array(cluster_indices)]
            centroid = cluster_embs.mean(axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            best_label = labels[cluster_indices[int(np.argmin(dists))]]

        result.append((best_label, total_count))

    result.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in result[:top_n]]


def _majority_vote(values: list[str]) -> str:
    """Return the most common value, defaulting to 'medium'."""
    if not values:
        return "medium"
    counter = Counter(values)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_signals(
    batch_results: list[dict],
    top_n: int = 15,
    threshold: float = 0.85,
) -> dict:
    """Merge signal dicts from multiple batches into one ranked result.

    Parameters
    ----------
    batch_results:
        List of signal dicts as returned by
        :func:`engine.extract.extract_signals_batch`.
    top_n:
        Maximum number of entries to retain per signal type.
    threshold:
        Cosine-similarity threshold for deduplication clustering.

    Returns
    -------
    dict
        Aggregated and ranked signal dict with the same schema as the
        extraction output.
    """
    # Collect all entries per signal type
    pain_points: list[dict] = []
    desired_outcomes: list[dict] = []
    purchase_triggers: list[dict] = []
    objections: list[dict] = []
    switching_triggers: list[dict] = []
    decision_factors: list[str] = []
    deal_breakers: list[str] = []
    friction_values: list[str] = []

    for batch in batch_results:
        pain_points.extend(batch.get("pain_points", []))
        desired_outcomes.extend(batch.get("desired_outcomes", []))
        purchase_triggers.extend(batch.get("purchase_triggers", []))
        objections.extend(batch.get("objections", []))
        switching_triggers.extend(batch.get("switching_triggers", []))
        decision_factors.extend(batch.get("decision_factors_ranked", []))
        deal_breakers.extend(batch.get("deal_breakers", []))
        friction_values.append(batch.get("friction_tolerance", "medium"))

    return {
        "pain_points": _merge_keyed_signals(
            pain_points, text_key="signal", threshold=threshold, top_n=top_n,
        ),
        "desired_outcomes": _merge_keyed_signals(
            desired_outcomes, text_key="outcome", threshold=threshold, top_n=top_n,
        ),
        "purchase_triggers": _merge_keyed_signals(
            purchase_triggers, text_key="trigger", threshold=threshold, top_n=top_n,
        ),
        "objections": _merge_keyed_signals(
            objections, text_key="objection", threshold=threshold, top_n=top_n,
        ),
        "switching_triggers": _merge_keyed_signals(
            switching_triggers, text_key="reason", threshold=threshold, top_n=top_n,
        ),
        "decision_factors_ranked": _merge_string_lists(decision_factors, top_n=top_n),
        "deal_breakers": _merge_string_lists(deal_breakers, top_n=top_n),
        "friction_tolerance": _majority_vote(friction_values),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demonstrate aggregation with synthetic batch results.
    batch_a: dict[str, Any] = {
        "pain_points": [
            {"signal": "Battery drains too quickly", "frequency": 5, "emotional_intensity": 0.8},
            {"signal": "Bluetooth connection drops frequently", "frequency": 3, "emotional_intensity": 0.7},
        ],
        "desired_outcomes": [
            {"outcome": "Longer battery life", "priority": 0.9},
        ],
        "purchase_triggers": [
            {"trigger": "Positive word of mouth", "context": "Friend recommendation"},
        ],
        "objections": [
            {"objection": "Too expensive for the features", "severity": 0.7},
        ],
        "switching_triggers": [
            {"from_product": "Sony WF-1000XM4", "reason": "Bulky fit", "threshold": "comfort"},
        ],
        "decision_factors_ranked": ["Sound quality", "Battery life", "Price"],
        "deal_breakers": ["Poor noise cancellation", "Uncomfortable fit"],
        "friction_tolerance": "medium",
    }

    batch_b: dict[str, Any] = {
        "pain_points": [
            {"signal": "Battery life is too short", "frequency": 4, "emotional_intensity": 0.75},
            {"signal": "App is buggy and crashes", "frequency": 2, "emotional_intensity": 0.6},
        ],
        "desired_outcomes": [
            {"outcome": "Better battery performance", "priority": 0.85},
            {"outcome": "Stable companion app", "priority": 0.6},
        ],
        "purchase_triggers": [
            {"trigger": "Recommendation from a friend", "context": "Social proof"},
        ],
        "objections": [
            {"objection": "Price is too high", "severity": 0.65},
        ],
        "switching_triggers": [
            {"from_product": "AirPods Pro", "reason": "Poor Android support", "threshold": "compatibility"},
        ],
        "decision_factors_ranked": ["Battery life", "Sound quality", "Comfort"],
        "deal_breakers": ["Bad noise cancellation"],
        "friction_tolerance": "low",
    }

    print("Aggregating 2 synthetic batches …")
    aggregated = aggregate_signals([batch_a, batch_b])
    print(json.dumps(aggregated, indent=2, default=str))
