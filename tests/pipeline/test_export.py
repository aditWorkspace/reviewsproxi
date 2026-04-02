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
