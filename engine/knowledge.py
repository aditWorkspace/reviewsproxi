"""
Persona Knowledge Store — Persistent Training Memory

Each persona accumulates learned knowledge from review data uploads.
This isn't a one-time extraction — it's an additive knowledge base that grows
with every CSV upload and persists forever.

Structure per persona:
  data/personas/{persona_id}/
    config.json          — Base persona configuration
    trained_knowledge.json — Accumulated signals, patterns, and context
    training_log.json    — Record of all training sessions
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.extract import extract_all_signals
from engine.aggregate import aggregate_signals


PERSONAS_DIR = Path("data/personas")

KNOWLEDGE_SYNTHESIS_PROMPT = """You are a behavioral analyst merging new review data into an existing persona knowledge base.

EXISTING PERSONA CONFIG:
{persona_config}

EXISTING TRAINED KNOWLEDGE (accumulated from previous training sessions):
{existing_knowledge}

NEW SIGNALS EXTRACTED FROM LATEST REVIEW BATCH ({n_reviews} reviews):
{new_signals}

Your job is to produce an UPDATED trained knowledge document that merges the new signals
into the existing knowledge. This is ADDITIVE — don't lose previously learned patterns.

Rules:
1. If new signals reinforce existing patterns, increase their confidence/weight
2. If new signals contradict existing patterns, note the tension but keep both
3. If new signals reveal entirely new patterns, add them
4. Maintain a running count of total reviews this persona has been trained on
5. Keep the top behavioral rules updated — if new data suggests a rule should change, update it
6. The voice_sample should evolve to incorporate newly discovered language patterns

Output valid JSON with this structure:
{{
  "total_reviews_trained_on": <int>,
  "last_trained": "<ISO timestamp>",
  "training_sessions": <int>,
  "core_pain_points": [
    {{"signal": "<description>", "confidence": <0-1>, "frequency": "high/medium/low", "source_count": <int>}}
  ],
  "core_desired_outcomes": [
    {{"outcome": "<description>", "priority": "primary/secondary/tertiary", "confidence": <0-1>}}
  ],
  "purchase_triggers": [
    {{"trigger": "<description>", "context": "<when this happens>", "confidence": <0-1>}}
  ],
  "objections": [
    {{"objection": "<description>", "severity": "blocking/major/minor", "confidence": <0-1>}}
  ],
  "switching_patterns": [
    {{"from": "<competitor/category>", "reason": "<why they switch>", "threshold": "<what breaks them>"}}
  ],
  "decision_factors_ranked": ["<factor1>", "<factor2>", "..."],
  "deal_breakers_confirmed": [
    {{"deal_breaker": "<description>", "times_seen": <int>, "confidence": <0-1>}}
  ],
  "behavioral_patterns": [
    {{"pattern": "<specific testable behavior>", "confidence": <0-1>, "source_count": <int>}}
  ],
  "emotional_vocabulary": ["<words and phrases this persona type actually uses>"],
  "price_sensitivity_details": {{
    "max_acceptable_price": "<range or description>",
    "free_tier_requirement": "<how important>",
    "price_comparison_behavior": "<how they compare>"
  }},
  "trust_signals": [
    {{"signal": "<what builds trust>", "strength": <0-1>}}
  ],
  "friction_patterns": [
    {{"friction": "<what causes frustration>", "severity": <0-1>, "typical_reaction": "<what they do>"}}
  ],
  "competitive_context": {{
    "alternatives_considered": ["<products they compare>"],
    "switching_cost_tolerance": "<low/medium/high>",
    "loyalty_factors": ["<what keeps them>"]
  }},
  "evolved_voice_sample": "<updated voice sample incorporating new language patterns>",
  "updated_behavioral_rules": [
    "<specific, testable behavioral rules derived from ALL training data>"
  ],
  "updated_decision_weights": {{
    "<factor>": <0-1 weight, all must sum to 1.0>
  }},
  "updated_emotional_profile": {{
    "baseline_patience": <0-1>,
    "trust_starting_point": <0-1>,
    "frustration_decay": <0-1>
  }},
  "raw_signal_archive": {{
    "pain_points_all": [],
    "triggers_all": [],
    "objections_all": []
  }}
}}
"""


def get_persona_dir(persona_id: str) -> Path:
    """Get or create the directory for a persona's persistent data."""
    # Check if it's an old-style single JSON file and migrate
    old_file = PERSONAS_DIR / f"{persona_id}.json"
    persona_dir = PERSONAS_DIR / persona_id

    if old_file.exists() and not persona_dir.exists():
        persona_dir.mkdir(parents=True, exist_ok=True)
        old_file.rename(persona_dir / "config.json")

    persona_dir.mkdir(parents=True, exist_ok=True)
    return persona_dir


def _find_config_path(persona_dir: Path) -> Path | None:
    """Return the config file inside a persona directory (config.json or persona.json)."""
    for name in ("config.json", "persona.json"):
        p = persona_dir / name
        if p.exists():
            return p
    return None


def load_persona_config(persona_id: str) -> dict:
    """Load a persona's base configuration."""
    persona_dir = get_persona_dir(persona_id)
    config_path = persona_dir / "config.json"

    if not config_path.exists():
        # Try persona.json (pipeline output format)
        alt = persona_dir / "persona.json"
        if alt.exists():
            with open(alt) as f:
                return json.load(f)
        # Check old-style flat files
        for f in PERSONAS_DIR.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
                if data.get("id") == persona_id:
                    return data
        raise FileNotFoundError(f"No config found for persona '{persona_id}'")

    with open(config_path) as f:
        return json.load(f)


def load_trained_knowledge(persona_id: str) -> dict | None:
    """Load a persona's accumulated trained knowledge, or None if untrained."""
    persona_dir = get_persona_dir(persona_id)
    knowledge_path = persona_dir / "trained_knowledge.json"

    if not knowledge_path.exists():
        return None

    with open(knowledge_path) as f:
        return json.load(f)


def load_training_log(persona_id: str) -> list[dict]:
    """Load the training session log for a persona."""
    persona_dir = get_persona_dir(persona_id)
    log_path = persona_dir / "training_log.json"

    if not log_path.exists():
        return []

    with open(log_path) as f:
        return json.load(f)


def save_trained_knowledge(persona_id: str, knowledge: dict) -> Path:
    """Save updated trained knowledge for a persona."""
    persona_dir = get_persona_dir(persona_id)
    knowledge_path = persona_dir / "trained_knowledge.json"

    with open(knowledge_path, "w") as f:
        json.dump(knowledge, f, indent=2, default=str)

    return knowledge_path


def append_training_log(persona_id: str, session: dict) -> None:
    """Append a training session record to the log."""
    persona_dir = get_persona_dir(persona_id)
    log_path = persona_dir / "training_log.json"

    log = load_training_log(persona_id)
    log.append(session)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)


def train_persona_on_reviews(
    persona_id: str,
    reviews: list[dict],
    source_name: str = "uploaded_csv",
    client=None,
    progress_callback: Any = None,
) -> dict:
    """
    Train a persona on a batch of reviews. This is the main entry point.

    1. Extract signals from reviews (LLM-powered)
    2. Aggregate signals across batches
    3. Merge with existing trained knowledge (LLM-powered synthesis)
    4. Save updated knowledge permanently
    5. Log the training session

    Returns the updated trained knowledge dict.
    """
    if client is None:
        from engine.llm import get_client
        client = get_client()

    persona_config = load_persona_config(persona_id)
    existing_knowledge = load_trained_knowledge(persona_id)

    # Step 1: Extract signals from reviews
    if progress_callback:
        progress_callback("Extracting behavioral signals from reviews...", 0.1)

    batch_results = extract_all_signals(
        reviews=reviews,
        category=persona_config.get("segment", {}).get("context", "general"),
        client=client,
        batch_size=30,
    )

    if progress_callback:
        progress_callback("Aggregating signals across batches...", 0.4)

    # Step 2: Aggregate
    aggregated = aggregate_signals(batch_results)

    if progress_callback:
        progress_callback("Synthesizing knowledge into persona...", 0.6)

    # Step 3: Merge with existing knowledge via LLM
    prompt = KNOWLEDGE_SYNTHESIS_PROMPT.format(
        persona_config=json.dumps(persona_config, indent=2),
        existing_knowledge=json.dumps(existing_knowledge, indent=2) if existing_knowledge else "No existing knowledge — this is the first training session.",
        new_signals=json.dumps(aggregated, indent=2),
        n_reviews=len(reviews),
    )

    from engine.llm import MODEL
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.choices[0].message.content

    # Parse JSON from response
    import re
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        response_text = json_match.group(1)

    try:
        updated_knowledge = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to salvage — find the outermost braces
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            updated_knowledge = json.loads(response_text[start:end])
        else:
            raise ValueError("Failed to parse knowledge synthesis response")

    if progress_callback:
        progress_callback("Saving trained knowledge...", 0.9)

    # Step 4: Save
    save_trained_knowledge(persona_id, updated_knowledge)

    # Step 5: Log
    session_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "n_reviews": len(reviews),
        "signals_extracted": {
            "pain_points": len(aggregated.get("pain_points", [])),
            "desired_outcomes": len(aggregated.get("desired_outcomes", [])),
            "purchase_triggers": len(aggregated.get("purchase_triggers", [])),
            "objections": len(aggregated.get("objections", [])),
            "deal_breakers": len(aggregated.get("deal_breakers", [])),
        },
    }
    append_training_log(persona_id, session_record)

    if progress_callback:
        progress_callback("Training complete!", 1.0)

    return updated_knowledge


def get_full_persona_context(persona_id: str) -> dict:
    """
    Load the FULL persona context for agent runs — base config + trained knowledge.
    This is what gets injected into the agent prompt.
    """
    config = load_persona_config(persona_id)
    knowledge = load_trained_knowledge(persona_id)
    training_log = load_training_log(persona_id)

    context = {
        "config": config,
        "trained_knowledge": knowledge,
        "training_summary": {
            "total_sessions": len(training_log),
            "total_reviews": sum(s.get("n_reviews", 0) for s in training_log),
            "last_trained": training_log[-1]["timestamp"] if training_log else None,
        },
        "is_trained": knowledge is not None,
    }

    # If trained, overlay learned behavioral rules and weights onto config
    if knowledge:
        if knowledge.get("updated_behavioral_rules"):
            config["behavioral_rules"] = knowledge["updated_behavioral_rules"]
        if knowledge.get("updated_decision_weights"):
            config["decision_weights"] = knowledge["updated_decision_weights"]
        if knowledge.get("updated_emotional_profile"):
            config["emotional_profile"] = knowledge["updated_emotional_profile"]
        if knowledge.get("evolved_voice_sample"):
            config["voice_sample"] = knowledge["evolved_voice_sample"]
        if knowledge.get("deal_breakers_confirmed"):
            config["deal_breakers"] = [
                db["deal_breaker"] for db in knowledge["deal_breakers_confirmed"]
            ]

    return context


def parse_csv_reviews(csv_content: str | bytes) -> list[dict]:
    """
    Parse a CSV of reviews into the standard review dict format.
    Flexible — handles various column naming conventions.
    """
    import csv
    import io

    if isinstance(csv_content, bytes):
        csv_content = csv_content.decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(csv_content))

    # Column name mapping — handles various CSV formats
    COLUMN_MAP = {
        # rating
        "rating": "rating", "stars": "rating", "star_rating": "rating",
        "score": "rating", "review_rating": "rating", "overall": "rating",
        # text
        "text": "text", "body": "text", "review_text": "text",
        "review_body": "text", "content": "text", "review": "text",
        "review_content": "text", "comment": "text", "reviewtext": "text",
        # title
        "title": "title", "summary": "title", "review_title": "title",
        "headline": "title", "review_headline": "title", "review_summary": "title",
        # product
        "asin": "asin", "product_id": "asin", "product_asin": "asin",
        "productid": "asin",
        # product name
        "product_name": "product_name", "product_title": "product_name",
        "product": "product_name", "item": "product_name", "name": "product_name",
        # category
        "category": "product_category", "product_category": "product_category",
        # date
        "date": "review_date", "review_date": "review_date", "timestamp": "review_date",
        # helpful
        "helpful_vote": "helpful_vote", "helpful": "helpful_vote",
        "helpful_votes": "helpful_vote", "upvotes": "helpful_vote",
        # verified
        "verified_purchase": "verified_purchase", "verified": "verified_purchase",
    }

    reviews = []
    for row in reader:
        review = {}
        for csv_col, value in row.items():
            normalized = csv_col.strip().lower().replace(" ", "_")
            standard_key = COLUMN_MAP.get(normalized, normalized)
            review[standard_key] = value

        # Convert rating to float if present
        if "rating" in review:
            try:
                review["rating"] = float(review["rating"])
            except (ValueError, TypeError):
                review["rating"] = 0

        # Skip empty reviews
        text = review.get("text", "") or ""
        if len(text.strip()) < 20:
            continue

        reviews.append(review)

    return reviews


def list_all_personas() -> list[dict]:
    """List all personas with their training status."""
    personas = []

    if not PERSONAS_DIR.exists():
        return personas

    # Check directory-style personas
    for d in PERSONAS_DIR.iterdir():
        cfg_path = _find_config_path(d) if d.is_dir() else None
        if d.is_dir() and cfg_path:
            with open(cfg_path) as f:
                config = json.load(f)
            knowledge = load_trained_knowledge(config["id"])
            log = load_training_log(config["id"])
            personas.append({
                "config": config,
                "is_trained": knowledge is not None,
                "training_sessions": len(log),
                "total_reviews_trained": sum(s.get("n_reviews", 0) for s in log),
            })

    # Check flat JSON files (old format)
    for f in PERSONAS_DIR.glob("*.json"):
        with open(f) as fh:
            config = json.load(fh)
        pid = config.get("id", f.stem)
        # Skip if already found as directory
        if any(p["config"].get("id") == pid for p in personas):
            continue
        personas.append({
            "config": config,
            "is_trained": False,
            "training_sessions": 0,
            "total_reviews_trained": 0,
        })

    return personas
