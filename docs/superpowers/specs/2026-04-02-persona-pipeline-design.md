# Persona Training Pipeline — Design Spec
**Date:** 2026-04-02  
**Status:** Approved  
**Target:** College Student persona (Phase 1), College Tech Entrepreneur (Phase 2)

---

## Overview

A staged, checkpointed pipeline that collects Amazon product reviews, extracts behavioral signals via DeepSeek V3.2, auto-optimizes clusters, and exports a dual-format persona spec (JSON + Markdown) ready to drop into the Proxi AI MVP repo.

The pipeline lives entirely in `pipeline.py` + `pipeline/` and does not touch the existing `main.py` agent runner. The existing `data/personas/college_tech_entrepreneur.json` and all agent code remain unchanged.

---

## Goals

1. Produce a grounded `college_student` persona spec from real Amazon review data
2. Export both a machine-readable JSON (for agent injection + RAG) and a human-readable Markdown (for prompt review and system prompt injection)
3. Switch all LLM calls from Anthropic to OpenRouter + DeepSeek V3.2
4. Make every stage resumable — LLM spend is never duplicated on re-runs
5. Keep the pipeline extensible to Phase 2 (college tech entrepreneur with cross-platform signals)

---

## Architecture

### New Files

```
pipeline.py                              ← CLI entry point (click group)
engine/llm.py                            ← OpenRouter/DeepSeek client factory
pipeline/
  collect.py                             ← Stage 1: HuggingFace + Playwright
  extract.py                             ← Stage 2: batched signal extraction
  cluster.py                             ← Stage 3: embed + auto-optimized cluster
  export.py                              ← Stage 4: persona synthesis + dual output
data/
  raw/reviews.jsonl                      ← checkpoint: collected reviews
  extracted/signals.jsonl               ← checkpoint: per-batch signals
  clusters/
    embeddings.npy                       ← checkpoint: sentence embeddings
    sweep.json                           ← cluster optimization scores
    clusters.json                        ← checkpoint: final cluster assignments
  personas/college_student/
    persona.json                         ← final structured persona spec
    persona.md                           ← final narrative context doc
    rag_index.jsonl                      ← review snippets keyed by review_id
```

### Modified Files

| File | Change |
|------|--------|
| `engine/extract.py` | Replace `anthropic.Anthropic()` with `engine/llm.get_client()` |
| `engine/knowledge.py` | Same client swap + `client.messages` → `client.chat.completions` |
| `engine/persona_builder.py` | Same client swap |
| `config.yaml` | Add `openrouter`, `scraper`, and `clustering` config blocks |
| `.env.example` | Swap `ANTHROPIC_API_KEY` → `OPENROUTER_API_KEY` |

### Unchanged

`agent/`, `insights/`, `storage/`, `app.py`, `main.py` — the agent runner is untouched. It will consume the new persona spec via the existing `load_persona_config()` path.

---

## LLM Client (`engine/llm.py`)

Uses the `openai` SDK pointed at OpenRouter. All pipeline stages and existing engines import `get_client()` from here.

```python
from openai import OpenAI
import os

def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

MODEL = "deepseek/deepseek-v3.2"
```

Call signature change across all three engines:
- `client.messages.create(model=..., system=..., messages=[...])` (Anthropic)
- → `client.chat.completions.create(model=MODEL, messages=[{"role": "system", ...}, {"role": "user", ...}])` (OpenAI-compatible)
- Response text: `message.content[0].text` → `response.choices[0].message.content`

---

## Stage 1: Collect (`pipeline/collect.py`)

**HuggingFace pull**

Dataset: `McAuley-Lab/Amazon-Reviews-2023`  
Categories: existing `config.yaml` list + `raw_review_Clothing_Shoes_and_Jewelry`, `raw_review_Sports_and_Outdoors`  
Filters: verified purchase only, `min_review_length: 50` chars  
Cap: `max_reviews_per_category: 5000`  
Output fields per review: `source`, `category`, `rating`, `text`, `timestamp`, `asin`, `reviewer_id`

**Playwright scraper (gap-fill)**

Targets student-specific Amazon search queries defined in `config.yaml`:

```yaml
scraper:
  target_queries:
    - "college backpack"
    - "student ID wallet"
    - "laptop stand desk"
    - "college dorm supplies"
    - "noise cancelling headphones student"
  max_reviews_per_query: 200
  headless: true
```

Flow per query: search Amazon → collect top 5 product ASINs → paginate reviews up to limit.  
Deduplication: by `asin + reviewer_id` hash against existing `reviews.jsonl`.  
Rate limiting: randomized delay 1.5–3.5s between requests.  
CAPTCHA/throttling: if Playwright detects a CAPTCHA or gets a non-200 response, it logs the ASIN, skips it, and continues — no crash. A `data/raw/scrape_errors.jsonl` file records skipped ASINs for manual review.

**Checkpoint:** if `data/raw/reviews.jsonl` exists and `--force` not passed, stage is skipped.

**CLI:**
```
python pipeline.py collect [--force] [--hf-only] [--scrape-only]
```

---

## Stage 2: Extract (`pipeline/extract.py`)

Reads `data/raw/reviews.jsonl`, batches into groups of 30, sends each batch to DeepSeek with the existing `SIGNAL_SCHEMA` prompt from `engine/extract.py`.

**Resume logic:** count lines already in `data/extracted/signals.jsonl`, skip that many input batches. Output is written line-by-line as batches complete — a mid-run crash resumes from where it stopped.

**Signal schema (unchanged from existing `engine/extract.py`):**
```json
{
  "pain_points": [{"signal": "...", "frequency": int, "emotional_intensity": float}],
  "desired_outcomes": [{"outcome": "...", "priority": float}],
  "purchase_triggers": [{"trigger": "...", "context": "..."}],
  "objections": [{"objection": "...", "severity": float}],
  "switching_triggers": [{"from_product": "...", "reason": "...", "threshold": "..."}],
  "decision_factors_ranked": ["..."],
  "deal_breakers": ["..."],
  "friction_tolerance": "low | medium | high"
}
```

**CLI:**
```
python pipeline.py extract [--force] [--batch-size 30]
```

---

## Stage 3: Cluster (`pipeline/cluster.py`)

**Embedding**

Each batch's signals are flattened into a text blob (pain points + desired outcomes + key phrases concatenated) and embedded using `sentence-transformers/all-MiniLM-L6-v2`.  
Saved to `data/clusters/embeddings.npy`.

**Auto-optimized clustering**

Sweeps `n_clusters` over a configurable range (default 3–10) and scores each run with three metrics:

| Metric | What it measures | Default weight |
|--------|-----------------|----------------|
| Silhouette score | Cluster separation | 0.5 |
| Intra-cluster cosine similarity | Coherence within clusters | 0.3 |
| Inter-cluster centroid distance | Distinctiveness between clusters | 0.2 |

All metrics computed locally (sklearn + numpy, no LLM calls). The `n_clusters` with the highest weighted score wins. Full sweep results saved to `data/clusters/sweep.json` for inspection.

`config.yaml` additions:
```yaml
clustering:
  sweep_range: [3, 10]
  scoring_weights:
    silhouette: 0.5
    intra_similarity: 0.3
    inter_distance: 0.2
```

`--n-clusters N` flag overrides the sweep entirely.

**Output (`data/clusters/clusters.json`):**
```json
{
  "chosen_n_clusters": 6,
  "cluster_0": {
    "label_hint": null,
    "member_count": 12,
    "representative_batches": [...],
    "aggregate_signals": {
      "top_pain_points": [...],
      "top_desired_outcomes": [...],
      "top_deal_breakers": [...],
      "dominant_friction_tolerance": "low"
    }
  }
}
```

`label_hint` is null here — DeepSeek names the cluster in Stage 4. The `chosen_n_clusters` value and top scoring metrics for the winning configuration are stored directly in `clusters.json` at the root level for full transparency:

```json
{
  "chosen_n_clusters": 6,
  "sweep_winner": {
    "silhouette": 0.61,
    "intra_similarity": 0.74,
    "inter_distance": 0.43,
    "combined_score": 0.634
  },
  "cluster_0": { ... }
}
```

**CLI:**
```
python pipeline.py cluster [--force] [--n-clusters N]
```

---

## Stage 4: Export (`pipeline/export.py`)

**Trait extraction (per cluster)**

Send each cluster's `aggregate_signals` + representative review snippets to DeepSeek. Extract:
- Trait label + description
- Key phrases (verbatim from reviews)
- Frequency score (`member_count / total_batches`)
- Emotional tone tag (`frustrated`, `cautious`, `aspirational`, etc.)

**Persona synthesis (full)**

Send all cluster traits + signals to DeepSeek for a single persona synthesis call. Produces the full behavioral profile matching the existing persona schema.

**JSON output** (`data/personas/college_student/persona.json`)

Extends existing persona schema with a `traits` array for RAG:
```json
{
  "id": "college_student",
  "label": "College Student",
  "traits": [
    {
      "label": "Durability Concern",
      "description": "Frequently reports products breaking within weeks of purchase",
      "key_phrases": ["snapped after a week", "fell apart", "cheap material"],
      "frequency": 0.38,
      "tone": "frustrated",
      "representative_review_ids": ["rev_0042", "rev_0189"]
    }
  ],
  "segment": {...},
  "goals": [...],
  "behavioral_rules": [...],
  "decision_weights": {...},
  "emotional_profile": {...},
  "deal_breakers": [...],
  "voice_sample": "..."
}
```

**Markdown output** (`data/personas/college_student/persona.md`)

Auto-generated narrative from the JSON. Sections:
1. Overview
2. Key Traits (one subsection per trait: description + sample quotes)
3. Behavioral Patterns
4. Decision Priorities
5. Deal Breakers
6. Voice Sample

**RAG index** (`data/personas/college_student/rag_index.jsonl`)

One line per representative review snippet. Each entry includes `review_id`, `trait_label`, `text`, and `tone` so the agent can retrieve the correct context by trait label at runtime. If a cluster's `label_hint` is null (e.g. batch failed or was empty), the export step assigns a fallback label `cluster_{idx}` and logs a warning — it does not halt the pipeline.

**CLI:**
```
python pipeline.py export [--force] [--persona-id college_student]
python pipeline.py run-all   ← runs all 4 stages in sequence
```

---

## Config Changes (`config.yaml`)

```yaml
openrouter:
  model: "deepseek/deepseek-v3.2"

data:
  amazon_categories:
    - "raw_review_Electronics"
    - "raw_review_Software"
    - "raw_review_Office_Products"
    - "raw_review_Computers"
    - "raw_review_Clothing_Shoes_and_Jewelry"   # new
    - "raw_review_Sports_and_Outdoors"           # new
  max_reviews_per_category: 5000
  min_review_length: 50
  verified_only: true

scraper:
  target_queries:
    - "college backpack"
    - "student ID wallet"
    - "laptop stand desk"
    - "college dorm supplies"
    - "noise cancelling headphones student"
  max_reviews_per_query: 200
  headless: true

clustering:
  sweep_range: [3, 10]
  scoring_weights:
    silhouette: 0.5
    intra_similarity: 0.3
    inter_distance: 0.2
```

---

## Environment

`.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Phase 2 Notes (College Tech Entrepreneur)

G2 data is noisier and more abstract — not easily mapped to product usage alone. Phase 2 will require:
- Cross-platform signals: G2 reviews + GitHub activity tone + Hacker News comment patterns
- Additional scraper targets: G2 product pages for dev tools (Vercel, Supabase, Stripe, etc.)
- Separate persona ID (`college_tech_entrepreneur`) fed into the same pipeline with different `target_queries` and an expanded signal schema that captures intent signals (e.g. "evaluating for production" vs "side project")
- The existing `data/personas/college_tech_entrepreneur.json` serves as a strong prior / seed config

Phase 2 is a config + query change, not an architecture change. The pipeline handles it natively.

Placeholder added to `config.yaml` for Phase 2 so switching personas is a one-line change:

```yaml
# Phase 2 — uncomment to activate
# active_persona: college_tech_entrepreneur
# scraper:
#   target_queries:
#     - "vercel review developer"
#     - "supabase postgres developer"
#     - "stripe payments startup"
#     - "github copilot student"
#   g2_targets:
#     - "vercel"
#     - "supabase"
#     - "linear"
#     - "notion"
```

---

## Success Criteria

- `data/personas/college_student/persona.json` is valid, loads via `load_persona_config()`, and works with the existing agent runner
- `data/personas/college_student/persona.md` is readable as a standalone system prompt context doc
- Pipeline resumes correctly from any stage after a crash
- All LLM calls route through OpenRouter/DeepSeek V3.2
- Existing `main.py` commands (`run`, `train`, `analyze`) are unaffected
