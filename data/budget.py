"""Source budget controller.

Enforces a per-source cap and a total-review target so no single platform
dominates the training corpus.  Pure functions — no Streamlit coupling.
"""

from __future__ import annotations

MIN_REVIEWS = 2_000
MAX_REVIEWS = 15_000
DEFAULT_TOTAL = 8_000
MAX_FRACTION = 0.40   # no single source may exceed 40 % of total


def source_cap(target_total: int, max_fraction: float = MAX_FRACTION) -> int:
    """Absolute maximum a single source may contribute."""
    return int(target_total * max_fraction)


def source_remaining(
    source_type: str,
    reviews: list[dict],
    target_total: int,
    max_fraction: float = MAX_FRACTION,
) -> int:
    """How many more reviews *source_type* is allowed to add given current corpus."""
    current = sum(1 for r in reviews if r.get("source_type") == source_type)
    cap = source_cap(target_total, max_fraction)
    return max(0, cap - current)


def source_fractions(reviews: list[dict]) -> dict[str, float]:
    """Return fraction of total reviews contributed by each source_type."""
    total = len(reviews)
    if total == 0:
        return {}
    counts: dict[str, int] = {}
    for r in reviews:
        src = r.get("source_type", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return {src: cnt / total for src, cnt in counts.items()}


def budget_summary(
    reviews: list[dict],
    target_total: int,
    max_fraction: float = MAX_FRACTION,
) -> dict:
    """Full budget status dict for UI display."""
    total = len(reviews)
    counts: dict[str, int] = {}
    for r in reviews:
        src = r.get("source_type", "unknown")
        counts[src] = counts.get(src, 0) + 1

    per_source = {}
    for src, cnt in counts.items():
        cap = source_cap(target_total, max_fraction)
        per_source[src] = {
            "count": cnt,
            "cap": cap,
            "remaining": max(0, cap - cnt),
            "pct": cnt / total if total else 0.0,
            "at_cap": cnt >= cap,
        }

    return {
        "total": total,
        "target": target_total,
        "progress_pct": min(1.0, total / target_total) if target_total else 0.0,
        "meets_minimum": total >= MIN_REVIEWS,
        "per_source": per_source,
    }
