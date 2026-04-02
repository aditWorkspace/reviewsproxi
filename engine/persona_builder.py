"""LLM-powered persona synthesis.

Turns clustered review data into rich, structured buyer personas using
Claude as the generative backbone.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from engine.cluster import build_reviewer_profiles, cluster_reviewers

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert B2B buyer-persona researcher. Given a set of product \
reviews from a behavioural cluster and aggregate signals about that cluster, \
you must synthesise a single, highly specific buyer persona.

Return ONLY valid JSON (no markdown fences) matching this schema exactly:

{
  "label": "<short human-readable name, e.g. 'Budget-Conscious IT Manager'>",
  "segment": {
    "company_size": "<e.g. '50-200 employees'>",
    "role": "<job title or function>",
    "tech_savviness": "<low | medium | high>",
    "price_sensitivity": "<low | medium | high>",
    "risk_tolerance": "<low | medium | high>"
  },
  "goals": ["<goal 1>", "<goal 2>", "..."],
  "constraints": ["<constraint 1>", "..."],
  "decision_weights": {
    "price": <float>,
    "features": <float>,
    "reliability": <float>,
    "support": <float>,
    "brand": <float>
  },
  "behavioral_rules": [
    "<rule 1 – must be testable: contain an action verb or a number>",
    "..."
  ],
  "emotional_profile": {
    "baseline_patience": <float 0-1>,
    "trust_starting_point": <float 0-1>,
    "frustration_decay": <float 0-1>
  },
  "deal_breakers": ["<concrete, binary condition>", "..."],
  "voice_sample": "<1-2 paragraph quote in the persona's authentic voice, with specific details>"
}

Rules:
- decision_weights values MUST sum to exactly 1.0.
- behavioral_rules must each contain an action verb or a number.
- deal_breakers must be concrete and binary (clearly true or false).
- voice_sample must reference specific product details or scenarios.
- Provide at least 3 goals, 2 constraints, 4 behavioral_rules, and 2 deal_breakers.
"""


def _build_user_prompt(
    cluster_reviews: list[dict],
    cluster_signals: dict,
    segment_label: str,
) -> str:
    """Compose the user-turn content for the persona synthesis call."""
    # Truncate reviews to avoid blowing the context window.
    sampled = cluster_reviews[:30]
    reviews_text = "\n---\n".join(
        f"Rating: {r.get('rating', 'N/A')}\n{r.get('text', '')}"
        for r in sampled
    )

    signals_text = json.dumps(cluster_signals, indent=2, default=str)

    return (
        f"Segment label: {segment_label}\n\n"
        f"## Cluster signals\n{signals_text}\n\n"
        f"## Representative reviews ({len(sampled)} shown)\n{reviews_text}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_persona(
    cluster_reviews: list[dict],
    cluster_signals: dict,
    segment_label: str,
    client,
) -> dict:
    """Use Claude to generate a structured persona from cluster data.

    Parameters
    ----------
    cluster_reviews:
        A list of review dicts representative of the cluster.
    cluster_signals:
        Aggregate statistics for the cluster (e.g. avg_rating,
        price_mentions, quality_mentions, etc.).
    segment_label:
        A short human-readable label for the segment (e.g.
        ``"price_sensitive_power_users"``).
    client:
        An initialised ``anthropic.Anthropic`` client instance.

    Returns
    -------
    dict
        The fully structured persona including a generated ``id``.
    """
    user_content = _build_user_prompt(cluster_reviews, cluster_signals, segment_label)

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

    # Tolerate markdown fences just in case.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    persona: dict[str, Any] = json.loads(raw_text)
    persona["id"] = str(uuid.uuid4())

    return persona


def build_personas_from_signals(
    reviews: list[dict],
    signals: dict,
    n_personas: int = 5,
    client=None,
) -> list[dict]:
    """Full pipeline: reviews -> clusters -> LLM-synthesised personas.

    Parameters
    ----------
    reviews:
        Flat list of review dicts.  Each must have ``"reviewer_id"``,
        ``"text"``, ``"rating"``, and optionally ``"category"``.
    signals:
        Global-level signals dict that will be forwarded to the LLM as
        additional context for every cluster.
    n_personas:
        Number of persona clusters to create.
    client:
        An ``anthropic.Anthropic`` instance.  If *None*, a default client
        is constructed (requires ``ANTHROPIC_API_KEY`` in the environment).

    Returns
    -------
    list[dict]
        One structured persona dict per cluster.
    """
    if client is None:
        from engine.llm import get_client
        client = get_client()

    # 1. Group reviews by reviewer.
    reviews_by_reviewer: dict[str, list[dict]] = {}
    for review in reviews:
        rid = review.get("reviewer_id", "anonymous")
        reviews_by_reviewer.setdefault(rid, []).append(review)

    # 2. Build behavioural profiles and cluster them.
    profiles = build_reviewer_profiles(reviews_by_reviewer)
    cluster_result = cluster_reviewers(profiles, n_clusters=n_personas)
    clusters = cluster_result["clusters"]

    # 3. Synthesise a persona for each cluster.
    personas: list[dict] = []
    for label_idx, cluster_data in clusters.items():
        cluster_profiles = cluster_data["profiles"]
        representative_reviews = cluster_data["representative_reviews"]

        # Aggregate cluster-level signals.
        cluster_signals = {
            "cluster_id": label_idx,
            "member_count": len(cluster_profiles),
            "avg_rating": (
                sum(p["avg_rating"] for p in cluster_profiles) / len(cluster_profiles)
                if cluster_profiles
                else 0.0
            ),
            "avg_price_mentions": (
                sum(p["price_mentions"] for p in cluster_profiles) / len(cluster_profiles)
                if cluster_profiles
                else 0.0
            ),
            "avg_quality_mentions": (
                sum(p["quality_mentions"] for p in cluster_profiles) / len(cluster_profiles)
                if cluster_profiles
                else 0.0
            ),
            "avg_comparison_mentions": (
                sum(p["comparison_mentions"] for p in cluster_profiles) / len(cluster_profiles)
                if cluster_profiles
                else 0.0
            ),
            "avg_emotional_valence": (
                sum(p["emotional_valence"] for p in cluster_profiles) / len(cluster_profiles)
                if cluster_profiles
                else 0.0
            ),
            **signals,
        }

        segment_label = f"cluster_{label_idx}"

        persona = synthesize_persona(
            cluster_reviews=representative_reviews,
            cluster_signals=cluster_signals,
            segment_label=segment_label,
            client=client,
        )
        personas.append(persona)

    return personas
