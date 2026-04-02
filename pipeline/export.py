"""Stage 4: Trait extraction, persona synthesis, and dual-format export."""
from __future__ import annotations

import json
from pathlib import Path

from engine.llm import get_client, MODEL

CLUSTERS_PATH = Path("data/clusters/clusters.json")

_TRAIT_SYSTEM = """\
You are a behavioral analyst. Given a cluster of behavioral signals from product reviews,
identify the dominant trait this cluster represents.

Return ONLY valid JSON (no markdown fences):
{
  "label": "<short trait name, e.g. 'Durability Concern'>",
  "description": "<1-2 sentences describing this trait>",
  "key_phrases": ["<verbatim phrase 1>", "<verbatim phrase 2>", "<verbatim phrase 3>"],
  "tone": "<one of: frustrated, cautious, aspirational, pragmatic, skeptical, enthusiastic>"
}"""

_PERSONA_SYSTEM = """\
You are an expert buyer-persona researcher. Given behavioral traits from product reviews for
college students, synthesize a complete, grounded buyer persona.

Return ONLY valid JSON (no markdown fences) matching this schema exactly:
{
  "id": "college_student",
  "label": "College Student",
  "segment": {
    "company_size": "N/A",
    "role": "Student",
    "tech_savviness": "<low|medium|high>",
    "price_sensitivity": "<low|medium|high|very_high>",
    "risk_tolerance": "<low|medium|high>",
    "age_range": "18-24",
    "context": "<1 sentence>"
  },
  "goals": [{"goal": "<string>", "priority": <int 1-5>}],
  "constraints": ["<string>"],
  "decision_weights": {"<factor>": <float, all sum to 1.0>},
  "behavioral_rules": ["<rule with action verb or number>"],
  "emotional_profile": {
    "baseline_patience": <float 0-1>,
    "trust_starting_point": <float 0-1>,
    "frustration_decay": <float 0-1>
  },
  "deal_breakers": ["<concrete, binary condition>"],
  "voice_sample": "<1-2 paragraph quote in authentic student voice>",
  "comparison_products": ["<product category comparisons>"],
  "browsing_patterns": {
    "typical_session_length_minutes": <int>,
    "pages_before_decision": <int>,
    "tab_behavior": "<string>",
    "device": "<string>",
    "time_of_day": "<string>"
  }
}

Rules: decision_weights must sum to 1.0; provide at least 4 goals, 3 constraints,
6 behavioral_rules, 3 deal_breakers."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_trait(cluster_id: str, cluster_data: dict, client) -> dict:
    """Ask DeepSeek to label a cluster as a behavioral trait."""
    signals_text = json.dumps(cluster_data["aggregate_signals"], indent=2)
    sample_phrases: list[str] = []
    for batch in cluster_data.get("representative_batches", [])[:2]:
        sample_phrases.extend(pp["signal"] for pp in batch.get("pain_points", [])[:3])
        sample_phrases.extend(do["outcome"] for do in batch.get("desired_outcomes", [])[:2])

    user_content = (
        f"Cluster ID: {cluster_id}\n\n"
        f"Aggregate signals:\n{signals_text}\n\n"
        f"Sample phrases:\n" + "\n".join(f"- {p}" for p in sample_phrases[:10])
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _TRAIT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    trait = json.loads(_strip_fences(response.choices[0].message.content))
    trait["cluster_id"] = cluster_id
    return trait


def synthesize_persona(traits: list[dict], clusters: dict, client) -> dict:
    """Synthesize full persona from all cluster traits."""
    total = sum(
        clusters[k]["member_count"] for k in clusters if k.startswith("cluster_")
    )
    for trait in traits:
        cid = trait["cluster_id"]
        trait["frequency"] = round(clusters[cid]["member_count"] / total, 3) if total else 0.0

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": _PERSONA_SYSTEM},
            {"role": "user", "content": f"Traits from {total} batches:\n\n{json.dumps(traits, indent=2)}"},
        ],
    )
    return json.loads(_strip_fences(response.choices[0].message.content))


def build_rag_index(traits: list[dict], clusters: dict) -> list[dict]:
    """Build RAG index entries from representative review snippets."""
    entries: list[dict] = []
    for trait in traits:
        cid = trait["cluster_id"]
        cluster_data = clusters.get(cid, {})
        for b_idx, batch in enumerate(cluster_data.get("representative_batches", [])):
            for pp_idx, pp in enumerate(batch.get("pain_points", [])[:3]):
                entries.append({
                    "review_id": f"{cid}_b{b_idx}_pp{pp_idx}",
                    "trait_label": trait["label"],
                    "text": pp["signal"],
                    "tone": trait.get("tone", "unknown"),
                })
            for do_idx, do in enumerate(batch.get("desired_outcomes", [])[:2]):
                entries.append({
                    "review_id": f"{cid}_b{b_idx}_do{do_idx}",
                    "trait_label": trait["label"],
                    "text": do["outcome"],
                    "tone": trait.get("tone", "unknown"),
                })
    return entries


def generate_markdown(persona: dict, traits: list[dict]) -> str:
    """Generate a narrative Markdown context doc from the persona JSON."""
    seg = persona.get("segment", {})
    lines = [
        f"# {persona.get('label', 'Persona')} — Persona Context Document",
        "",
        "## Overview",
        "",
        seg.get("context", ""),
        "",
        f"- **Age range:** {seg.get('age_range', 'N/A')}",
        f"- **Price sensitivity:** {seg.get('price_sensitivity', 'N/A')}",
        f"- **Tech savviness:** {seg.get('tech_savviness', 'N/A')}",
        "",
        "## Key Traits",
        "",
    ]

    for trait in sorted(traits, key=lambda t: -t.get("frequency", 0)):
        freq_pct = f"{trait.get('frequency', 0):.0%}"
        lines += [
            f"### {trait['label']} (frequency: {freq_pct})",
            "",
            trait.get("description", ""),
            "",
            f"**Tone:** {trait.get('tone', 'N/A')}",
            "",
            "**Sample phrases:**",
        ]
        for phrase in trait.get("key_phrases", []):
            lines.append(f'- "{phrase}"')
        lines.append("")

    lines += ["## Behavioral Patterns", ""]
    for rule in persona.get("behavioral_rules", []):
        lines.append(f"- {rule}")
    lines.append("")

    lines += ["## Decision Priorities", ""]
    weights = persona.get("decision_weights", {})
    for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
        lines.append(f"- **{factor}:** {weight:.0%}")
    lines.append("")

    lines += ["## Deal Breakers", ""]
    for db in persona.get("deal_breakers", []):
        lines.append(f"- {db}")
    lines.append("")

    lines += ["## Voice Sample", "", f'> {persona.get("voice_sample", "")}', ""]

    return "\n".join(lines)


def run_export(persona_id: str = "college_student", force: bool = False) -> None:
    """Run Stage 4: extract traits, synthesize persona, write all outputs."""
    persona_dir = Path(f"data/personas/{persona_id}")
    json_path = persona_dir / "persona.json"

    if json_path.exists() and not force:
        print(f"[export] {json_path} already exists. Use --force to regenerate.")
        return

    persona_dir.mkdir(parents=True, exist_ok=True)

    with open(CLUSTERS_PATH) as f:
        clusters = json.load(f)

    client = get_client()
    cluster_keys = [k for k in clusters if k.startswith("cluster_")]
    traits: list[dict] = []

    for cid in cluster_keys:
        try:
            trait = extract_trait(cid, clusters[cid], client)
        except Exception as e:
            print(f"[export] Warning: trait extraction failed for {cid}: {e}")
            trait = {
                "label": cid,
                "description": "Trait extraction failed — using cluster ID as fallback label.",
                "key_phrases": [],
                "tone": "unknown",
                "cluster_id": cid,
            }
        traits.append(trait)

    persona = synthesize_persona(traits, clusters, client)
    persona["traits"] = [
        {
            "label": t["label"],
            "description": t["description"],
            "key_phrases": t["key_phrases"],
            "frequency": t.get("frequency", 0.0),
            "tone": t["tone"],
            "representative_review_ids": [
                f"{t['cluster_id']}_b0_pp0",
                f"{t['cluster_id']}_b0_pp1",
            ],
        }
        for t in traits
    ]

    with open(json_path, "w") as f:
        json.dump(persona, f, indent=2)

    md_path = persona_dir / "persona.md"
    with open(md_path, "w") as f:
        f.write(generate_markdown(persona, traits))

    rag_path = persona_dir / "rag_index.jsonl"
    with open(rag_path, "w") as f:
        for entry in build_rag_index(traits, clusters):
            f.write(json.dumps(entry) + "\n")

    print(f"[export] Written:\n  {json_path}\n  {md_path}\n  {rag_path}")
