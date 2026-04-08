"""
Proxi AI — Persona Training Dashboard
Build rich demographic persona context files from real user data.

Run:  source .env && streamlit run train.py
Need: OPENROUTER_API_KEY
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# Load .env automatically so users don't need to `source .env` manually
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import numpy as np
import streamlit as st
import functools

# ---------------------------------------------------------------------------
# Worker-mode guard
# ---------------------------------------------------------------------------
# When PROXI_WORKER_MODE=1, this module is imported by worker.py to access
# pipeline functions.  All Streamlit calls are skipped so the import doesn't
# crash outside a Streamlit context.

_WORKER_MODE = bool(os.getenv("PROXI_WORKER_MODE"))

# st.cache_resource is a Streamlit-only construct.  In worker mode, replace it
# with a plain lru_cache so cached functions still work correctly.
_cache_resource = (lambda *a, **k: (a[0] if a and callable(a[0]) else lambda f: f)) if _WORKER_MODE else st.cache_resource

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

if not _WORKER_MODE:
    st.set_page_config(
        page_title="Proxi — Persona Training",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0b0d14; }
[data-testid="stMainBlockContainer"] { max-width: 980px; padding-top: 1.5rem; }
[data-testid="stSidebar"] { display: none; }

/* ── Typography ────────────────────────────────────────────── */
h1, h2, h3 { letter-spacing: -0.02em; }

/* ── Cards ──────────────────────────────────────────────────── */
.card {
    background: #13151f;
    border: 1px solid #22253a;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s;
}
.card-accent { border-left: 3px solid #6d5ff7; }

/* ── Queue cards ────────────────────────────────────────────── */
.qcard {
    background: #13151f;
    border: 1px solid #22253a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.qcard:hover { border-color: #3a3d5a; box-shadow: 0 2px 12px rgba(0,0,0,0.3); }
.qcard-running { border-left: 3px solid #facc15; background: #14130b; }
.qcard-done    { border-left: 3px solid #4ade80; background: #0d1410; }
.qcard-failed  { border-left: 3px solid #f87171; background: #140d0d; }
.qcard-pending { border-left: 3px solid #475569; }

.qcard-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: #e2e8f0;
    line-height: 1.3;
}
.qcard-meta {
    font-size: 0.76rem;
    color: #64748b;
    margin-top: 3px;
}

/* ── Status badges ──────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.s-running { background: #2d2400; color: #facc15; border: 1px solid #854d0e; }
.s-done    { background: #0d2016; color: #4ade80; border: 1px solid #166534; }
.s-failed  { background: #200d0d; color: #f87171; border: 1px solid #7f1d1d; }
.s-pending { background: #1a1d27; color: #94a3b8; border: 1px solid #334155; }

/* ── Queue stats bar ────────────────────────────────────────── */
.queue-stats {
    display: flex;
    gap: 1.5rem;
    padding: 0.75rem 0;
    margin-bottom: 0.5rem;
}
.qstat { text-align: center; }
.qstat-num  { font-size: 1.4rem; font-weight: 800; line-height: 1; }
.qstat-lbl  { font-size: 0.7rem; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Animated pulse for running ─────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.pulse { animation: pulse-glow 2s ease-in-out infinite; }

/* ── Step indicator ─────────────────────────────────────────── */
.step-row {
    display: flex;
    gap: 0.35rem;
    margin-bottom: 1.5rem;
    align-items: center;
    flex-wrap: wrap;
}
.step {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-weight: 600;
    white-space: nowrap;
}
.step-done    { background: #0d2016; color: #4ade80; border: 1px solid #166534; }
.step-active  { background: #1e1650; color: #a78bfa; border: 1px solid #6d28d9; }
.step-pending { background: #13151f; color: #374151; border: 1px solid #1e2130; }
.step-arrow   { color: #2a2d3a; font-size: 0.65rem; }

/* ── Product cards ──────────────────────────────────────────── */
.product-card {
    background: #13151f;
    border: 1px solid #22253a;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.4rem;
    transition: border-color 0.15s;
}
.confidence-high { border-left: 3px solid #4ade80; }
.confidence-med  { border-left: 3px solid #fbbf24; }
.confidence-low  { border-left: 3px solid #f87171; }

/* ── Live log ───────────────────────────────────────────────── */
.log-box {
    background: #07090f;
    border: 1px solid #1a1d2a;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.77rem;
    max-height: 260px;
    overflow-y: auto;
    line-height: 1.65;
}

/* ── Quality badges ─────────────────────────────────────────── */
.badge { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 4px;
         font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; }
.badge-strong   { background: #0d2016; color: #4ade80; }
.badge-good     { background: #0d1a0d; color: #86efac; }
.badge-adequate { background: #1f1600; color: #fbbf24; }
.badge-weak     { background: #1f0d0d; color: #f87171; }

/* ── Subreddit chips ────────────────────────────────────────── */
.sub-chip {
    display: inline-block;
    background: #16133a;
    border: 1px solid #3b1f7c;
    border-radius: 20px;
    padding: 0.18rem 0.65rem;
    font-size: 0.78rem;
    color: #c4b5fd;
    margin: 0.18rem;
}

/* ── Dividers ───────────────────────────────────────────────── */
hr { border-color: #1e2130 !important; margin: 1.25rem 0 !important; }

/* ── Streamlit overrides ────────────────────────────────────── */
[data-testid="stExpander"] > div:first-child {
    border-radius: 8px !important;
    border-color: #22253a !important;
}
button[kind="primary"] { transition: opacity 0.15s !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants  (accessible in both UI and worker mode)
# ---------------------------------------------------------------------------

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# Cost tiers — designed to stay under ~$0.50 per demographic run
#
#   CHEAP  (high-volume repeated calls): deepseek/deepseek-chat
#          ~$0.14/M in, $0.28/M out — used for bulk extraction batches
#
#   MID    (thread filter, many small calls): qwen/qwen-plus
#          ~$0.40/M — fast, cheap, good enough for relevance scoring
#
#   SMART  (one-shot intelligence + synthesis): claude-sonnet-4
#          ~$3/M in, $15/M out — reserved for the two calls where
#          reasoning quality directly shapes all downstream output
#
MODEL           = "deepseek/deepseek-chat"          # bulk extraction (high volume)
MODEL_FILTER    = "qwen/qwen-plus"                   # thread relevance filter
MODEL_SMART     = "anthropic/claude-sonnet-4"        # intelligence + synthesis
MODEL_INTELLIGENCE = MODEL_SMART                     # keep existing references working

# ---------------------------------------------------------------------------
# Sentiment helpers for implied-rating inference (unrated sources)
# ---------------------------------------------------------------------------

_SENT_POS: set[str] = {
    "love","great","excellent","amazing","awesome","fantastic","wonderful",
    "best","happy","pleased","impressed","recommend","intuitive","easy",
    "seamless","delight","perfect","superb","fast","reliable","smooth",
}
_SENT_NEG: set[str] = {
    "hate","terrible","awful","worst","horrible","frustrating","disappointed",
    "annoying","useless","broken","poor","bad","confusing","clunky","ugly",
    "painful","nightmare","regret","slow","buggy","crash","crashes","laggy",
}
_NEGATION: set[str] = {
    "not","no","never","barely","hardly",
    "isn't","wasn't","doesn't","don't","won't","can't","couldn't","didn't",
}


def _infer_rating(text: str) -> float:
    """Infer a 1–5 star equivalent from comment sentiment.

    Used for Reddit / HN comments that have no explicit star rating so the
    quality checker can correctly report the negative/positive review balance.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    pos = neg = 0
    for i, t in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in _NEGATION
        if t in _SENT_POS:
            if negated: neg += 1
            else:        pos += 1
        elif t in _SENT_NEG:
            if negated: pos += 1
            else:        neg += 1
    total = pos + neg
    valence = (pos - neg) / total if total else 0.0
    if valence <= -0.5: return 1.0
    if valence <= -0.1: return 2.0
    if valence <=  0.1: return 3.0
    if valence <=  0.5: return 4.0
    return 5.0


# ---------------------------------------------------------------------------
# Semantic scoring helpers (reuses sentence-transformers already installed)
# ---------------------------------------------------------------------------

@_cache_resource(show_spinner=False)
def _load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _persona_embedding(demographic_desc: str):
    """Return a normalised embedding for the demographic description (cached)."""
    model = _load_embed_model()
    import numpy as np
    emb = model.encode([demographic_desc], normalize_embeddings=True)
    return emb[0]


def _batch_semantic_scores(texts: list[str], persona_emb) -> list[float]:
    """Cosine similarity of each text against the persona embedding.

    Both embeddings are L2-normalised so dot product == cosine similarity.
    Runs in a single batch — no per-comment LLM calls.
    """
    if not texts:
        return []
    import numpy as np
    model = _load_embed_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return (embs @ persona_emb).tolist()

HF_CATEGORIES: dict[str, str] = {
    "All_Beauty": "All Beauty",
    "Amazon_Fashion": "Amazon Fashion",
    "Appliances": "Appliances",
    "Arts_Crafts_and_Sewing": "Arts, Crafts & Sewing",
    "Automotive": "Automotive",
    "Baby_Products": "Baby Products",
    "Beauty_and_Personal_Care": "Beauty & Personal Care",
    "Books": "Books",
    "CDs_and_Vinyl": "CDs & Vinyl",
    "Cell_Phones_and_Accessories": "Cell Phones & Accessories",
    "Clothing_Shoes_and_Jewelry": "Clothing, Shoes & Jewelry",
    "Digital_Music": "Digital Music",
    "Electronics": "Electronics",
    "Grocery_and_Gourmet_Food": "Grocery & Gourmet Food",
    "Health_and_Household": "Health & Household",
    "Home_and_Kitchen": "Home & Kitchen",
    "Industrial_and_Scientific": "Industrial & Scientific",
    "Kindle_Store": "Kindle Store",
    "Movies_and_TV": "Movies & TV",
    "Musical_Instruments": "Musical Instruments",
    "Office_Products": "Office Products",
    "Patio_Lawn_and_Garden": "Patio, Lawn & Garden",
    "Pet_Supplies": "Pet Supplies",
    "Software": "Software",
    "Sports_and_Outdoors": "Sports & Outdoors",
    "Subscription_Boxes": "Subscription Boxes",
    "Tools_and_Home_Improvement": "Tools & Home Improvement",
    "Toys_and_Games": "Toys & Games",
    "Video_Games": "Video Games",
}

# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------

def project_dir(pid: str) -> Path:
    return PROJECTS_DIR / pid

def reviews_path(pid: str) -> Path:
    return project_dir(pid) / "reviews.jsonl"

def signals_path(pid: str) -> Path:
    return project_dir(pid) / "signals.jsonl"

def clusters_path(pid: str) -> Path:
    return project_dir(pid) / "clusters.json"

def intelligence_path(pid: str) -> Path:
    return project_dir(pid) / "intelligence.json"

def outputs_dir(pid: str) -> Path:
    return project_dir(pid) / "outputs"

def _output_stem(pid: str) -> str:
    """Return a filename stem like 'disgo_consumer_app_user_hospitality_food_discovery_context'."""
    p = load_project(pid)
    if p:
        company_slug = re.sub(r"[^a-z0-9]+", "_", (p["name"] or "").lower()).strip("_")
        label_slug   = re.sub(r"[^a-z0-9]+", "_", (p.get("demographic_label") or "").lower()).strip("_")
        if company_slug and label_slug:
            return f"{company_slug}_{label_slug}"
        if company_slug:
            return f"{company_slug}_{pid}"
    return pid

def list_projects() -> list[dict]:
    out = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        m = d / "meta.json"
        if d.is_dir() and m.exists():
            out.append(json.loads(m.read_text()))
    return out

def load_project(pid: str) -> dict | None:
    m = project_dir(pid) / "meta.json"
    return json.loads(m.read_text()) if m.exists() else None

def save_project(p: dict) -> None:
    d = project_dir(p["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(p, indent=2))

def new_project(name: str, label: str, description: str, pid: str | None = None) -> dict:
    if pid is None:
        pid = f"{label.lower().replace(' ','_').replace('-','_')}_{uuid.uuid4().hex[:6]}"
    p = {
        "id": pid, "name": name,
        "demographic_label": label,
        "demographic_description": description,
        "created_at": datetime.utcnow().isoformat(),
        "review_count": 0,
        "review_sources": [],
        "stages": {
            "intelligence": "pending",
            "reviews": "empty",
            "signals": "pending",
            "clusters": "pending",
            "persona": "pending",
        },
    }
    save_project(p)
    outputs_dir(pid).mkdir(parents=True, exist_ok=True)
    return p

def count_reviews(pid: str) -> int:
    rp = reviews_path(pid)
    if not rp.exists(): return 0
    with open(rp) as f:
        return sum(1 for _ in f)

def load_all_reviews(pid: str) -> list[dict]:
    rp = reviews_path(pid)
    if not rp.exists(): return []
    return [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]

def append_reviews(pid: str, reviews: list[dict], source_label: str) -> int:
    """Append reviews, skipping exact text duplicates already in the corpus."""
    import hashlib
    rp = reviews_path(pid)

    # Build set of already-stored text hashes to prevent duplicates on rerun
    existing_hashes: set[str] = set()
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                try:
                    text = json.loads(line).get("text", "")
                    existing_hashes.add(hashlib.md5(text.encode()).hexdigest())
                except Exception:
                    pass

    new_reviews = []
    for r in reviews:
        h = hashlib.md5((r.get("text") or "").encode()).hexdigest()
        if h not in existing_hashes:
            new_reviews.append(r)
            existing_hashes.add(h)

    if new_reviews:
        with open(rp, "a") as f:
            for r in new_reviews:
                f.write(json.dumps(r) + "\n")

    total = count_reviews(pid)
    p = load_project(pid)
    p["review_count"] = total
    p["stages"]["reviews"] = "ready"
    p["review_sources"].append({
        "label": source_label,
        "count": len(new_reviews),
        "skipped_dupes": len(reviews) - len(new_reviews),
        "at": datetime.utcnow().isoformat()[:10],
    })
    save_project(p)
    return total

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _parse_keys() -> list[str]:
    raw = os.environ.get("OPENROUTER_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

def get_client():
    from openai import OpenAI
    keys = _parse_keys()
    if not keys:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=keys[0])

def get_clients():
    from openai import OpenAI
    keys = _parse_keys()
    if not keys:
        raise ValueError("OPENROUTER_API_KEY not set")
    return [OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k) for k in keys]

def llm(messages: list[dict], max_tokens: int = 4000, json_mode: bool = False, model: str | None = None) -> str:
    client = get_client()
    kwargs: dict = dict(model=model or MODEL, max_tokens=max_tokens, messages=messages)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()

def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first complete, balanced JSON object from arbitrary text.

    Unlike a greedy regex this walks forward counting braces so it stops
    exactly where the object ends, even if the LLM appended trailing prose
    or a second JSON block after the main response.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from an LLM response, handling fences and trailing content."""
    # 1. Try clean parse after stripping code fences
    try:
        return json.loads(strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Extract the first balanced JSON object (handles trailing LLM prose)
    candidate = _extract_first_json_object(raw) or _extract_first_json_object(strip_fences(raw))
    if candidate:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}

# ---------------------------------------------------------------------------
# Step 0 — Generate demographic intelligence
# ---------------------------------------------------------------------------
# Uses LLM (Sonnet) to map a demographic description to specific Amazon products
# and subreddits. This is the ONE step where LLM "creativity" is intentional —
# we need it to identify what products this demographic actually buys.
# Output: products (with accuracy estimates), subreddits, demographic_profile keywords.
# These keywords power the downstream filtering in Steps 1-2.

_INTELLIGENCE_PROMPT = """\
You are building a high-quality training dataset for a synthetic AI persona representing a specific demographic.

DEMOGRAPHIC: {description}

We are searching within the HuggingFace "McAuley-Lab/Amazon-Reviews-2023" dataset which has 571M reviews across these categories: {categories}

Your task has 3 parts. Return a single JSON object with keys: products, subreddits, demographic_profile.

━━━ PART 1: Targeted Amazon Products ━━━

Identify exactly 18 specific Amazon products where **90%+ of buyers ARE CURRENTLY this exact demographic** — not an adjacent one.

CRITICAL RULES:
- The buyer must BE THIS DEMOGRAPHIC right now, not preparing to become it or having been it before.
- Products for a DIFFERENT life stage that merely relates to this demographic are WRONG.
- Think: "Who is physically clicking 'Buy' on Amazon for this product?" — that person must be your demographic.

Examples for "college students (ages 18-22, currently enrolled)":
✓ GOOD: "College Ruled Composition Notebooks 6-Pack" — college students buy these for class
✓ GOOD: "Keurig K-Mini Single Serve Coffee Maker" — dorm room staple, bought by the student themselves
✓ GOOD: "Brita Standard Water Filter Pitcher" — students buy for dorm/apartment
✗ WRONG: "ACT/SAT Prep Book" — bought by/for HIGH SCHOOL students, not college students
✗ WRONG: "LSAT Prep" — bought by post-college/grad school applicants
✗ WRONG: "Baby Einstein toys" — bought by PARENTS, not babies
✗ WRONG: "iPhone 15" — everyone buys iPhones, not demographic-specific

Ask yourself for EACH product: "Would someone NOT in this demographic ever buy this?" If yes, lower the accuracy or drop it.

For each product, use one of these exact category keys: {categories}

{{
  "name": "exact Amazon product name as it would appear on Amazon",
  "category": "one of the category keys listed above (e.g. raw_review_Books)",
  "accuracy": <0.0-1.0, how confident that 90%+ of buyers ARE this demographic>,
  "review_keywords": ["5-8 words likely IN the review text for this product"],
  "context_keywords": ["3-5 words that appear in reviews WRITTEN BY this demographic, e.g. 'campus', 'dorm', 'semester'"],
  "exclusion_keywords": ["2-3 words indicating reviewer is NOT this demographic"],
  "why": "one sentence — WHO buys this and WHY it's specific to this demographic"
}}

Sort by accuracy descending.

━━━ PART 2: Subreddits ━━━

List 15 specific subreddits where the majority of active posters ARE this demographic (not just interested in the topic).

For each:
{{
  "name": "subredditname (no r/)",
  "relevance": <0.0-1.0>,
  "member_type": "what % and what kind of members from this demographic",
  "data_value": "what behavioral insights you'd get from posts here",
  "why": "one sentence"
}}

━━━ PART 3: Demographic Profile ━━━

{{
  "core_keywords": ["10-15 words appearing in content FROM or ABOUT this demographic"],
  "exclusion_keywords": ["5-8 words indicating NOT this demographic"],
  "age_range": "...",
  "primary_motivations": ["top 4 things this demographic cares about when buying"],
  "summary": "3 sentences — who they are, what drives their decisions, what frustrates them"
}}
"""

def generate_intelligence(pid: str, description: str, log_fn=None) -> dict:
    if log_fn: log_fn("info", f"Using Sonnet to map {description[:60]}... to specific products + subreddits")

    category_list = ", ".join(HF_CATEGORIES.keys())
    prompt = _INTELLIGENCE_PROMPT.format(description=description, categories=category_list)
    raw = llm([{"role": "user", "content": prompt}], max_tokens=5000, model=MODEL_INTELLIGENCE)

    data = _parse_llm_json(raw)

    # Save
    intelligence_path(pid).write_text(json.dumps(data, indent=2))

    p = load_project(pid)
    p["stages"]["intelligence"] = "complete"
    save_project(p)

    if log_fn:
        products = data.get("products", [])
        subs = data.get("subreddits", [])
        log_fn("ok", f"Found {len(products)} targeted products, {len(subs)} subreddits")
        for prod in products[:5]:
            log_fn("data", f"  [{prod.get('accuracy',0):.0%}] {prod.get('name','?')}")

    return data

def load_intelligence(pid: str) -> dict | None:
    p = intelligence_path(pid)
    return json.loads(p.read_text()) if p.exists() else None

# ---------------------------------------------------------------------------
# Step 1 — Pull Amazon reviews (product-targeted)
# ---------------------------------------------------------------------------
# Streams from HuggingFace McAuley-Lab/Amazon-Reviews-2023 (571M reviews).
# For each product identified in Step 0, matches reviews using keyword overlap:
#   - review_keywords: words likely in the review text for that product
#   - context_keywords: demographic-specific language (e.g. "dorm", "campus")
#   - exclusion_keywords: signals the reviewer is NOT in our demographic
# Each matched review gets tagged with product_accuracy from the intelligence phase.
# Amazon reviews are our HIGHEST quality signal — they're product-specific purchase
# feedback with ratings, verified purchase status, and detailed text.

def pull_amazon_reviews(
    pid: str,
    intelligence: dict,
    selected_products: list[str],
    max_per_product: int,
    target_total: int = 5_000,
    log_fn=None,
) -> int:
    from datasets import load_dataset

    products = [p for p in intelligence.get("products", []) if p["name"] in selected_products]
    if not products:
        return 0

    # Group products by category (strip "raw_review_" prefix if LLM included it)
    by_category: dict[str, list[dict]] = {}
    for prod in products:
        cat = prod.get("category", "")
        cat = re.sub(r"^raw_review_", "", cat)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(prod)

    demo_keywords = set(w.lower() for w in intelligence.get("demographic_profile", {}).get("core_keywords", []))
    total_added = 0

    for cat, cat_products in by_category.items():
        if cat not in HF_CATEGORIES:
            continue
        cat_label = HF_CATEGORIES[cat]
        if log_fn: log_fn("info", f"Streaming {cat_label} from HuggingFace...")

        # Enforce source budget — Amazon capped at 40% of target_total
        from data.budget import source_remaining
        amazon_remaining = source_remaining("amazon", load_all_reviews(pid), target_total)
        if amazon_remaining <= 0:
            if log_fn: log_fn("warn", "Amazon budget full — skipping remaining categories")
            break
        # Don't pull more per-product than the budget allows across all products
        effective_max = min(max_per_product, max(1, amazon_remaining // max(1, len(cat_products))))

        # Build keyword sets per product
        product_quota: dict[str, int] = {p["name"]: effective_max for p in cat_products}
        product_keywords: dict[str, tuple[set, set, set]] = {}
        for prod in cat_products:
            rk = set(w.lower() for kw in prod.get("review_keywords", []) for w in kw.split())
            ck = set(w.lower() for kw in prod.get("context_keywords", []) for w in kw.split())
            ek = set(w.lower() for kw in prod.get("exclusion_keywords", []) for w in kw.split())
            product_keywords[prod["name"]] = (rk, ck, ek)

        try:
            ds = load_dataset(
                "json",
                data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{cat}.jsonl",
                split="train", streaming=True,
            )
        except Exception as e:
            if log_fn: log_fn("warn", f"Could not load {cat_label}: {e}")
            continue

        scanned = 0
        cat_added = 0

        for row in ds:
            if all(v <= 0 for v in product_quota.values()):
                break

            text = (row.get("text", "") or "").strip()
            if len(text) < 40:
                continue

            text_lower = text.lower()
            scanned += 1

            # Match against each product
            for prod in cat_products:
                pname = prod["name"]
                if product_quota.get(pname, 0) <= 0:
                    continue

                rk, ck, ek = product_keywords[pname]

                # Exclusion check
                if any(w in text_lower for w in ek):
                    continue

                # Must match review keywords OR context keywords
                rk_match = sum(1 for w in rk if w in text_lower)
                ck_match = sum(1 for w in ck if w in text_lower)
                demo_match = sum(1 for w in demo_keywords if w in text_lower)

                if rk_match >= 1 or (ck_match + demo_match) >= 2:
                    rating = float(row.get("rating", row.get("overall", 3)) or 3)
                    append_reviews(pid, [{
                        "text": text,
                        "rating": rating,
                        "source": "amazon",
                        "source_type": "amazon",
                        "product": pname,
                        "product_accuracy": prod.get("accuracy", 0.5),
                        "category": cat_label,
                        "asin": row.get("asin", ""),
                    }], source_label=f"Amazon/{pname[:30]}")
                    product_quota[pname] -= 1
                    cat_added += 1
                    total_added += 1
                    break  # don't double-count a review for multiple products

            if scanned % 5000 == 0 and log_fn:
                left = sum(v for v in product_quota.values() if v > 0)
                log_fn("info", f"  Scanned {scanned:,} reviews, collected {cat_added} so far ({left} slots left)")

        if log_fn: log_fn("ok", f"  Done {cat_label}: collected {cat_added} reviews (scanned {scanned:,})")

    return total_added

# ---------------------------------------------------------------------------
# Step 2 — Reddit scraper (automated via JSON API)
# ---------------------------------------------------------------------------
# Reddit provides broader behavioral context but is NOISIER than Amazon:
#   - Comments are discussion-based, not purchase-focused reviews
#   - Demographic targeting relies on subreddit selection + thread filtering
#   - Comment-level scoring uses keyword heuristics (not verified purchases)
# For this reason, Reddit data gets a 0.7x discount in extraction weighting
# (see _process_batch). Reddit is still valuable because it captures:
#   - Unprompted opinions and frustrations
#   - Peer recommendations and social proof patterns
#   - Price sensitivity language in natural context
# Thread filtering: LLM scores thread titles for demographic relevance,
# then per-comment keyword scoring filters off-demographic noise.

REDDIT_HEADERS = {"User-Agent": "ProxiAI/1.0 (persona training pipeline)"}
REDDIT_MIN_COMMENT_LEN = 40

# Recurring weekly/daily thread titles to deduplicate (keep only the best instance)
_RECURRING_PREFIXES = [
    "gym story", "rant wednesday", "moronic monday", "training tuesday",
    "physique phriday", "victory sunday", "simple questions", "daily thread",
    "weekly thread", "daily discussion", "weekly discussion", "daily question",
    "megathread", "monthly",
]


def _is_recurring_title(title: str) -> bool:
    t = title.lower().strip()
    return any(t.startswith(p) for p in _RECURRING_PREFIXES)


def _normalize_title(title: str) -> str:
    """Collapse recurring thread titles to a canonical form for dedup."""
    t = re.sub(r"\s*[-—]\s*(january|february|march|april|may|june|july|august|september|october|november|december).*", "", title, flags=re.IGNORECASE)
    t = re.sub(r"\s*[-—]\s*\d{1,2}/\d{1,2}.*", "", t)
    t = re.sub(r"\s*[-—]\s*\d{4}.*", "", t)
    return t.strip().lower()


def scrape_subreddit_threads(
    subreddit: str,
    max_threads: int = 10,
    sort: str = "top",
    time_filter: str = "year",
    log_fn=None,
) -> list[dict]:
    """Fetch top threads from a subreddit, deduplicated.

    Returns list of {title, url, permalink, score, num_comments, author}.
    """
    import requests

    threads: list[dict] = []
    seen_titles: dict[str, dict] = {}  # normalized_title -> best thread
    after = None
    pages = 0
    max_pages = 6  # 25 per page * 6 = 150 threads scanned

    while pages < max_pages:
        url = f"https://old.reddit.com/r/{subreddit}/{sort}.json"
        params = {"t": time_filter, "limit": 25, "raw_json": 1}
        if after:
            params["after"] = after

        if log_fn: log_fn("info", f"  Fetching r/{subreddit}/{sort} (page {pages + 1})...")

        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if log_fn: log_fn("warn", f"  Failed to fetch r/{subreddit}: {e}")
            break

        posts = data.get("data", {}).get("children", [])
        if not posts:
            break

        for post in posts:
            pd = post.get("data", {})
            title = pd.get("title", "")
            num_comments = pd.get("num_comments", 0)
            permalink = pd.get("permalink", "")

            # Skip low-comment threads
            if num_comments < 10:
                continue

            norm = _normalize_title(title)
            thread_info = {
                "title": title,
                "permalink": permalink,
                "score": pd.get("score", 0),
                "num_comments": num_comments,
                "author": pd.get("author", ""),
                "is_recurring": _is_recurring_title(title),
            }

            # Dedup recurring threads: keep the one with most comments
            if _is_recurring_title(title):
                if norm in seen_titles:
                    if num_comments > seen_titles[norm]["num_comments"]:
                        seen_titles[norm] = thread_info
                else:
                    seen_titles[norm] = thread_info
            else:
                # Unique thread — always keep
                threads.append(thread_info)

        after = data.get("data", {}).get("after")
        if not after:
            break
        pages += 1
        time.sleep(2.0)  # rate limit

    # Add best instance of each recurring thread
    threads.extend(seen_titles.values())

    # Sort by comment count, take top N
    threads.sort(key=lambda t: -t["num_comments"])
    threads = threads[:max_threads]

    if log_fn:
        log_fn("ok", f"  Found {len(threads)} unique threads from r/{subreddit}")
        for t in threads[:5]:
            tag = " [recurring-best]" if t["is_recurring"] else ""
            log_fn("data", f"    [{t['num_comments']} comments] {t['title'][:60]}{tag}")

    return threads


def scrape_thread_comments(
    permalink: str,
    max_comments: int = 200,
    log_fn=None,
) -> list[dict]:
    """Fetch comments from a thread via JSON API. Gracefully handles threads
    with fewer comments than max_comments — just returns what's available."""
    import requests

    url = f"https://old.reddit.com{permalink}.json"
    params = {"limit": 500, "depth": 8, "sort": "top", "raw_json": 1}

    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=20)
        if resp.status_code == 429:
            time.sleep(10)
            resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                if log_fn: log_fn("warn", f"  Rate limited, skipping thread")
                return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if log_fn: log_fn("warn", f"  Failed to fetch thread: {e}")
        return []

    comments: list[dict] = []

    def _walk_comments(children: list):
        for child in children:
            if len(comments) >= max_comments:
                return
            if child.get("kind") != "t1":
                continue
            cd = child.get("data", {})
            body = (cd.get("body", "") or "").strip()
            score = cd.get("score", 1)
            author = cd.get("author", "")

            # Skip bots, deleted, short
            if author in ("AutoModerator", "[deleted]", ""):
                continue
            if len(body) < REDDIT_MIN_COMMENT_LEN:
                continue

            comments.append({
                "text": body,
                "upvotes": score,
                "author": author,
                "source_type": "reddit",
            })

            # Recurse into replies
            replies = cd.get("replies")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                _walk_comments(reply_children)

    # data[1] contains the comment listing
    if isinstance(data, list) and len(data) > 1:
        comment_children = data[1].get("data", {}).get("children", [])
        _walk_comments(comment_children)

    return comments


_THREAD_FILTER_PROMPT = """\
Score Reddit threads for relevance to a specific target demographic.

TARGET DEMOGRAPHIC: {demographic}
SUBREDDIT: r/{subreddit}

Score each thread on FOUR criteria (0–3 each, max 12 per thread):

CRITERION 1 — DEMOGRAPHIC FIT: What fraction of active commenters ARE the target demographic?
  0 = General audience (any age/background posts here)
  1 = Overlapping (meaningful overlap but not dominant)
  2 = Primarily this demographic
  3 = Almost exclusively this demographic

CRITERION 2 — PRODUCT/EXPERIENCE CONTEXT: Does the thread discuss buying, evaluating, or using relevant product types?
  0 = No product/service context
  1 = Tangential mention
  2 = Direct product category discussion
  3 = Active purchase evaluation or detailed usage comparison

CRITERION 3 — PROBLEM/DESIRE SPECIFICITY: Does it surface specific frustrations, needs, or behavioral signals?
  0 = Generic opinions or venting
  1 = Some specific details
  2 = Clear pain points or desires stated
  3 = Rich, specific behavioral signals with context

CRITERION 4 — FIRST-PERSON EXPERIENCE: Are commenters sharing personal experience vs. giving generic advice?
  0 = Theoretical or advice-giving only
  1 = Mix of personal and generic
  2 = Mostly personal accounts
  3 = Almost entirely first-person product/service experiences

Compute: relevance = (c1 + c2 + c3 + c4) / 12   (rounds to 2 decimal places)

Return a JSON array — ONE object per thread, omit threads where relevance < 0.40:
[
  {{"index": 0, "c1": 2, "c2": 3, "c3": 2, "c4": 1, "relevance": 0.67, "reason": "max 10 words"}},
  ...
]

THREADS:
{threads_json}
"""


def filter_threads_for_demographic(
    threads: list[dict],
    demographic_description: str,
    subreddit: str,
    log_fn=None,
) -> list[dict]:
    """Use LLM to score threads by demographic relevance, drop irrelevant ones."""
    if not threads:
        return []

    threads_for_prompt = [
        {"index": i, "title": t["title"], "comments": t["num_comments"], "score": t["score"]}
        for i, t in enumerate(threads)
    ]

    prompt = _THREAD_FILTER_PROMPT.format(
        demographic=demographic_description,
        subreddit=subreddit,
        threads_json=json.dumps(threads_for_prompt, indent=1),
    )

    try:
        raw = llm([{"role": "user", "content": prompt}], max_tokens=2000, model=MODEL_FILTER)
        cleaned = strip_fences(raw)
        # Handle case where LLM wraps in extra text
        m = re.search(r"\[[\s\S]*\]", cleaned)
        scored = json.loads(m.group(0) if m else cleaned)
    except Exception as e:
        if log_fn: log_fn("warn", f"  Thread filter failed ({e}) — keeping all threads")
        # Fallback: keep all with default relevance
        for t in threads:
            t["demographic_relevance"] = 0.5
        return threads

    # Map relevance scores back to threads.  The new rubric returns c1–c4 +
    # a pre-computed relevance, so we validate and recompute for safety.
    relevance_map: dict[int, float] = {}
    for item in scored:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            continue
        # Prefer pre-computed relevance; fall back to recomputing from criteria.
        if "relevance" in item:
            rel = float(item["relevance"])
        else:
            c_sum = sum(item.get(f"c{k}", 0) for k in range(1, 5))
            rel = round(c_sum / 12, 3)
        relevance_map[idx] = rel

    filtered = []
    for i, thread in enumerate(threads):
        rel = relevance_map.get(i, 0.0)
        thread["demographic_relevance"] = rel
        thread["rubric_scores"] = next(
            (item for item in scored if isinstance(item, dict) and item.get("index") == i),
            {},
        )
        if rel >= 0.40:   # lowered from 0.6 — rubric is more conservative
            filtered.append(thread)

    if log_fn:
        dropped = len(threads) - len(filtered)
        if dropped:
            log_fn("info", f"  Kept {len(filtered)}/{len(threads)} threads (dropped {dropped} with relevance < 0.40)")

    # Sort by relevance × comment count (highest signal density first)
    filtered.sort(key=lambda t: -(t["demographic_relevance"] * t["num_comments"]))
    return filtered


def score_comments_semantic_batch(
    comments: list[dict],
    persona_emb,
    exclusion_keywords: set[str],
) -> list[float]:
    """Batch-score comments by semantic similarity to the persona embedding.

    Replaces the old keyword heuristic (score = 0.5 + hits×0.1) which
    produced nearly identical scores for very different comments.

    Uses cosine similarity between L2-normalised embeddings:
      - All embeddings computed in one batch (fast, no per-comment LLM calls)
      - Hard exclusion filter: 2+ exclusion keyword hits → score capped at 0.15
      - Final score in [0.0, 1.0]
    """
    texts = [c.get("text", "") for c in comments]
    sims = _batch_semantic_scores(texts, persona_emb)

    scores = []
    for text, sim in zip(texts, sims):
        text_lower = text.lower()
        excl_hits = sum(1 for kw in exclusion_keywords if kw in text_lower)
        if excl_hits >= 2:
            scores.append(0.15)
        else:
            # Cosine similarity is already in [-1, 1]; shift to [0, 1]
            scores.append(max(0.0, min(1.0, (sim + 1) / 2)))
    return scores


def scrape_reddit_for_project(
    pid: str,
    subreddits: list[str],
    max_threads_per_sub: int = 10,
    max_comments_per_thread: int = 200,
    target_total: int = 5_000,
    log_fn=None,
) -> int:
    """Scrape Reddit with rubric-based thread filtering and semantic comment scoring."""
    from data.budget import source_remaining

    total_added = 0

    # Load demographic profile for filtering
    intel = load_intelligence(pid)
    demo_profile = (intel or {}).get("demographic_profile", {})
    demographic_desc = demo_profile.get("summary", "")
    if not demographic_desc:
        p = load_project(pid)
        demographic_desc = p.get("demographic_description", "general consumer")

    exclusion_keywords = set(w.lower() for w in demo_profile.get("exclusion_keywords", []))

    # Compute persona embedding once — reused for every comment in this session
    persona_emb = _persona_embedding(demographic_desc)

    for sub in subreddits:
        # Check budget before doing any work for this subreddit
        remaining = source_remaining("reddit", load_all_reviews(pid), target_total)
        if remaining <= 0:
            if log_fn: log_fn("warn", f"  Reddit budget full ({target_total * 0.4:.0f} cap reached) — stopping")
            break

        if log_fn: log_fn("info", f"Scraping r/{sub}… (budget remaining: {remaining})")

        # Step 1: Candidate threads
        threads = scrape_subreddit_threads(sub, max_threads=max_threads_per_sub * 2, log_fn=log_fn)

        # Step 2: Rubric-based thread filter (LLM)
        if threads and demographic_desc:
            if log_fn: log_fn("info", f"  Scoring threads with 4-criterion rubric…")
            threads = filter_threads_for_demographic(threads, demographic_desc, sub, log_fn=log_fn)
            threads = threads[:max_threads_per_sub]

        sub_comments = 0
        for i, thread in enumerate(threads):
            remaining = source_remaining("reddit", load_all_reviews(pid), target_total)
            if remaining <= 0:
                break

            thread_rel = thread.get("demographic_relevance", 0.5)
            if log_fn: log_fn("info", f"  Thread {i+1}/{len(threads)} [relevance {thread_rel:.2f}]: {thread['title'][:55]}…")

            comments = scrape_thread_comments(
                thread["permalink"],
                max_comments=max_comments_per_thread,
                log_fn=log_fn,
            )
            if not comments:
                continue

            # Step 3: Batch semantic scoring — one encoder pass per thread
            comment_scores = score_comments_semantic_batch(comments, persona_emb, exclusion_keywords)

            reviews = []
            for c, sem_score in zip(comments, comment_scores):
                # Weighted combination: thread relevance carries 60%, semantic 40%
                combined = round(0.6 * thread_rel + 0.4 * sem_score, 3)
                if combined < 0.35:
                    continue

                # Infer implied rating from sentiment (replaces hardcoded 3.0)
                implied = _infer_rating(c["text"])

                reviews.append({
                    "text": c["text"],
                    "rating": implied,
                    "source_type": "reddit",
                    "source": f"r/{sub}",
                    "subreddit": sub,
                    "thread_title": thread["title"],
                    "upvotes": c.get("upvotes", 1),
                    "product_accuracy": combined,
                    "thread_relevance": thread_rel,
                    "semantic_score": sem_score,
                })

            # Trim to budget
            reviews = reviews[:remaining]

            if reviews:
                append_reviews(pid, reviews, f"Reddit/r/{sub}/{thread['title'][:30]}")
                sub_comments += len(reviews)
                total_added += len(reviews)
                high_rel = sum(1 for r in reviews if r["product_accuracy"] >= 0.5)
                if log_fn: log_fn("data", f"    → {len(reviews)} comments ({high_rel} high-relevance, avg rating {sum(r['rating'] for r in reviews)/len(reviews):.1f}★)")

            time.sleep(3.0)  # respect Reddit rate limits

        if log_fn: log_fn("ok", f"  r/{sub}: {sub_comments} comments from {len(threads)} threads")

    return total_added

# ---------------------------------------------------------------------------
# Hacker News scraper
# ---------------------------------------------------------------------------

def scrape_hn_for_project(
    pid: str,
    queries: list[str],
    max_per_query: int = 250,
    target_total: int = 5_000,
    log_fn=None,
) -> int:
    """Scrape HN comments and score them semantically against the persona."""
    from data.sources.hackernews import scrape_hn_for_project as _hn_scrape
    from data.budget import source_remaining

    remaining = source_remaining("hackernews", load_all_reviews(pid), target_total)
    if remaining <= 0:
        if log_fn: log_fn("warn", "HN budget full — skipping")
        return 0

    # Load demographic description for semantic scoring
    intel = load_intelligence(pid)
    demo_profile = (intel or {}).get("demographic_profile", {})
    demographic_desc = demo_profile.get("summary", "")
    if not demographic_desc:
        p = load_project(pid)
        demographic_desc = p.get("demographic_description", "general consumer")

    exclusion_keywords = set(w.lower() for w in demo_profile.get("exclusion_keywords", []))
    persona_emb = _persona_embedding(demographic_desc)

    raw_comments = _hn_scrape(queries, max_per_query=max_per_query, min_upvotes=2, log_fn=log_fn)
    if not raw_comments:
        return 0

    # Batch semantic scoring
    sem_scores = score_comments_semantic_batch(raw_comments, persona_emb, exclusion_keywords)

    reviews = []
    for c, sem in zip(raw_comments, sem_scores):
        if sem < 0.35:
            continue
        reviews.append({
            "text": c["text"],
            "rating": _infer_rating(c["text"]),
            "source_type": "hackernews",
            "source": "hackernews",
            "story_title": c.get("story_title", ""),
            "upvotes": c.get("upvotes", 0),
            "product_accuracy": round(sem, 3),
            "semantic_score": round(sem, 3),
            "hn_id": c.get("hn_id", ""),
        })

    reviews = reviews[:remaining]
    if reviews:
        append_reviews(pid, reviews, "HackerNews")
        if log_fn: log_fn("ok", f"HN: {len(reviews)} comments added (avg sem score {sum(r['semantic_score'] for r in reviews)/len(reviews):.2f})")

    return len(reviews)


# ---------------------------------------------------------------------------
# Step 2c — App Store (Apple)
# ---------------------------------------------------------------------------

def scrape_appstore_for_project(
    pid: str,
    queries: list[str],
    max_per_app: int = 200,
    target_total: int = 5_000,
    log_fn=None,
) -> int:
    """Pull App Store reviews for apps matching the demographic's product list."""
    from data.sources.appstore import scrape_appstore
    from data.budget import source_remaining

    remaining = source_remaining("appstore", load_all_reviews(pid), target_total)
    if remaining <= 0:
        if log_fn: log_fn("warn", "App Store budget full — skipping")
        return 0

    p = load_project(pid)
    demographic_desc = p.get("demographic_description", "general consumer")
    exclusion_keywords: set[str] = set()
    intel = load_intelligence(pid)
    if intel:
        excl = intel.get("demographic_profile", {}).get("exclusion_keywords", [])
        exclusion_keywords = set(w.lower() for w in excl)
    persona_emb = _persona_embedding(demographic_desc)

    raw = scrape_appstore(queries, max_per_app=min(max_per_app, remaining), log_fn=log_fn)
    if not raw:
        return 0

    sem_scores = score_comments_semantic_batch(raw, persona_emb, exclusion_keywords)

    reviews = []
    for r, sem in zip(raw, sem_scores):
        if sem < 0.30:
            continue
        reviews.append({
            "text": r["text"],
            "rating": r.get("rating") or _infer_rating(r["text"]),
            "source_type": "appstore",
            "source": "appstore",
            "title": r.get("title", ""),
            "app_name": r.get("app_name", ""),
            "product_accuracy": round(sem, 3),
            "semantic_score": round(sem, 3),
        })

    reviews = reviews[:remaining]
    if reviews:
        append_reviews(pid, reviews, "App Store")
        if log_fn:
            log_fn("ok", f"App Store: {len(reviews)} reviews added")
    return len(reviews)


# ---------------------------------------------------------------------------
# Step 2d — Google Play Store
# ---------------------------------------------------------------------------

def scrape_playstore_for_project(
    pid: str,
    queries: list[str],
    max_per_app: int = 200,
    target_total: int = 5_000,
    log_fn=None,
) -> int:
    """Pull Google Play reviews for apps matching the demographic's product list."""
    from data.sources.playstore import scrape_playstore
    from data.budget import source_remaining

    remaining = source_remaining("playstore", load_all_reviews(pid), target_total)
    if remaining <= 0:
        if log_fn: log_fn("warn", "Play Store budget full — skipping")
        return 0

    p = load_project(pid)
    demographic_desc = p.get("demographic_description", "general consumer")
    exclusion_keywords: set[str] = set()
    intel = load_intelligence(pid)
    if intel:
        excl = intel.get("demographic_profile", {}).get("exclusion_keywords", [])
        exclusion_keywords = set(w.lower() for w in excl)
    persona_emb = _persona_embedding(demographic_desc)

    raw = scrape_playstore(queries, max_per_app=min(max_per_app, remaining), log_fn=log_fn)
    if not raw:
        return 0

    sem_scores = score_comments_semantic_batch(raw, persona_emb, exclusion_keywords)

    reviews = []
    for r, sem in zip(raw, sem_scores):
        if sem < 0.30:
            continue
        reviews.append({
            "text": r["text"],
            "rating": r.get("rating") or _infer_rating(r["text"]),
            "source_type": "playstore",
            "source": "playstore",
            "app_name": r.get("app_name", ""),
            "upvotes": r.get("upvotes", 0),
            "product_accuracy": round(sem, 3),
            "semantic_score": round(sem, 3),
        })

    reviews = reviews[:remaining]
    if reviews:
        append_reviews(pid, reviews, "Play Store")
        if log_fn:
            log_fn("ok", f"Play Store: {len(reviews)} reviews added")
    return len(reviews)


# ---------------------------------------------------------------------------
# Step 3 — CSV / custom review parsing
# ---------------------------------------------------------------------------

def parse_csv_reviews(raw: bytes, source_label: str) -> list[dict]:
    content = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(content))

    COLUMN_MAP = {
        "text": "text", "body": "text", "review_text": "text", "review_body": "text",
        "content": "text", "review": "text", "comment": "text", "description": "text",
        "rating": "rating", "stars": "rating", "star_rating": "rating",
        "score": "rating", "review_rating": "rating", "overall": "rating",
        "title": "title", "summary": "title", "review_title": "title",
        "product_name": "product", "product_title": "product", "product": "product",
        "category": "category", "product_category": "category",
        "source": "original_source",
    }

    reviews = []
    for row in reader:
        r: dict = {"source_type": "csv", "source": source_label}
        for col, val in row.items():
            key = COLUMN_MAP.get(col.strip().lower().replace(" ", "_"), col.lower().strip())
            r[key] = (val or "").strip()
        try:
            r["rating"] = float(r.get("rating", 3) or 3)
        except (ValueError, TypeError):
            r["rating"] = 3.0
        text = (r.get("text", "") or "").strip()
        if len(text) >= 30:
            reviews.append(r)

    return reviews

# ---------------------------------------------------------------------------
# Data quality analysis
# ---------------------------------------------------------------------------
# Computes a quality score from 0-8 points based on objective metrics:
#   +3 for 800+ reviews, +2 for 400+, +1 for 200+
#   +2 for 2+ sources (cross-platform validation)
#   +1 for balanced rating distribution (both positive and negative)
#   +1 for avg review length >= 120 chars (more detail = richer signals)
#   +1 for 5+ distinct products (breadth of coverage)
# Quality gates: "strong" (6+), "good" (4+), "adequate" (2+), "weak" (<2)
# Issues list flags specific problems: volume, source diversity, rating skew.

def analyze_quality(reviews: list[dict]) -> dict:
    if not reviews:
        return {
            "score": "no_data", "score_pts": 0, "total": 0, "avg_rating": 0,
            "avg_length": 0, "rating_distribution": {}, "source_breakdown": {},
            "n_products": 0, "top_products": {}, "low_rating_pct": 0,
            "high_rating_pct": 0, "n_sources": 0, "batches_estimate": 0,
            "issues": ["No reviews loaded yet."],
        }

    total = len(reviews)

    # For unrated sources (Reddit, HN) use the pre-computed implied rating stored
    # during scraping.  This ensures the rating distribution reflects real sentiment
    # instead of showing everything as 3★ (the old hardcoded default).
    def _effective_rating(r: dict) -> float:
        rt = r.get("rating")
        if rt is not None and float(rt) != 3.0:
            return float(rt)
        # If rating is missing or still the old hardcoded 3.0, infer from text
        src = r.get("source_type", "")
        if src in ("reddit", "hackernews"):
            return _infer_rating(r.get("text", ""))
        return float(rt) if rt is not None else 3.0

    ratings = [_effective_rating(r) for r in reviews]
    avg_rating = sum(ratings) / len(ratings)
    avg_len = sum(len(r.get("text", "")) for r in reviews) / total

    rating_dist = Counter(int(round(rt)) for rt in ratings)
    low_pct  = (rating_dist.get(1, 0) + rating_dist.get(2, 0)) / total
    high_pct = (rating_dist.get(4, 0) + rating_dist.get(5, 0)) / total

    source_dist = Counter(r.get("source_type", "unknown") for r in reviews)
    n_sources = len(source_dist)
    product_dist = Counter(r.get("product", "unknown") for r in reviews)

    # Source dominance check
    max_source = max(source_dist.values()) if source_dist else 0
    max_source_name = max(source_dist, key=source_dist.get) if source_dist else "unknown"
    max_source_pct = max_source / total if total else 0

    issues = []
    if total < 200:  issues.append(f"Low volume ({total} reviews) — need 300+ for reliable signal")
    if avg_len < 80: issues.append(f"Short reviews (avg {avg_len:.0f} chars) — may lack detail")
    if low_pct < 0.08: issues.append("Very few negative reviews — pain points may be underrepresented")
    if high_pct < 0.25: issues.append("Very few positive reviews — purchase triggers may be underrepresented")
    if n_sources < 2: issues.append("Single source — add Reddit or CSV for richer behavioral signal")
    if total > 0 and len(product_dist) < 3: issues.append("Narrow product coverage — diversify products for broader signal")
    if max_source_pct > 0.40:
        issues.append(f"Platform bias: {max_source_name} is {max_source_pct:.0%} of data (cap is 40%) — signals may reflect platform culture, not demographic behavior. Add other sources.")

    # Quality score
    score_pts = 0
    if total >= 800: score_pts += 3
    elif total >= 400: score_pts += 2
    elif total >= 200: score_pts += 1
    if n_sources >= 2: score_pts += 2
    if low_pct >= 0.08 and high_pct >= 0.25: score_pts += 1
    if avg_len >= 120: score_pts += 1
    if len(product_dist) >= 5: score_pts += 1

    score = "strong" if score_pts >= 6 else "good" if score_pts >= 4 else "adequate" if score_pts >= 2 else "weak"

    return {
        "score": score,
        "score_pts": score_pts,
        "total": total,
        "avg_rating": round(avg_rating, 2),
        "avg_length": int(avg_len),
        "rating_distribution": dict(rating_dist),
        "source_breakdown": dict(source_dist),
        "n_products": len(product_dist),
        "top_products": dict(product_dist.most_common(5)),
        "low_rating_pct": round(low_pct, 3),
        "high_rating_pct": round(high_pct, 3),
        "n_sources": n_sources,
        "issues": issues,
        "batches_estimate": (total + 29) // 30,
    }

# ---------------------------------------------------------------------------
# Pipeline: Signal extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline: Signal extraction
# ---------------------------------------------------------------------------
# Sends batches of ~30 reviews to the LLM (via engine/extract.py) to extract
# structured behavioral signals: pain_points, desired_outcomes, purchase_triggers,
# deal_breakers, objections, friction_tolerance.
#
# Key design choices:
#   - Source-balanced batching: interleaves Amazon/Reddit/CSV reviews so each
#     batch gets a mix, preventing platform bias in any single extraction.
#   - Demographic weighting: high-accuracy reviews (>= 0.85) are duplicated
#     in the batch so the LLM sees them more, amplifying on-target signal.
#   - Reddit discount: Reddit reviews get 0.7x on their accuracy score before
#     the weighting threshold, so fewer Reddit comments get the 2x boost.
#   - Crash-safe resume: signals are written to signals.jsonl incrementally,
#     so if extraction crashes at batch 80/142, rerun picks up at batch 81.
#   - Parallel execution: uses ThreadPoolExecutor with multiple API keys.
#
# IMPORTANT: The LLM extraction returns qualitative labels (pain point text,
# trigger descriptions) — NOT numeric scores. Any "frequency" or "intensity"
# values from the LLM are DISCARDED during clustering. The real frequencies
# come from counting how often a signal appears across batches (see
# _count_with_confidence in the clustering step).

EXTRACTION_TIMEOUT_MINUTES = 25
MAX_PARALLEL = 25

def run_extraction(pid: str, batch_size: int, force: bool, log_fn=None) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from engine.extract import extract_signals_batch

    import random

    reviews = load_all_reviews(pid)
    if not reviews:
        raise ValueError("No reviews to extract from.")

    sp = signals_path(pid)
    sp.parent.mkdir(parents=True, exist_ok=True)

    if force and sp.exists():
        sp.unlink()

    done = 0
    if sp.exists():
        with open(sp) as f:
            done = sum(1 for _ in f)

    # Source-balanced batching: interleave sources so each batch has a mix
    # This prevents platform bias (e.g., 30 Amazon reviews in a row)
    by_source: dict[str, list[dict]] = {}
    for r in reviews:
        src = r.get("source_type", "unknown")
        by_source.setdefault(src, []).append(r)

    # Interleave: round-robin across sources, then batch
    interleaved: list[dict] = []
    source_iters = {k: iter(v) for k, v in by_source.items()}
    while source_iters:
        exhausted = []
        for src, it in source_iters.items():
            val = next(it, None)
            if val is None:
                exhausted.append(src)
            else:
                interleaved.append(val)
        for src in exhausted:
            del source_iters[src]

    # Use interleaved order for new extractions, but keep original order for already-done batches
    if done == 0:
        reviews_ordered = interleaved
    else:
        reviews_ordered = reviews  # don't reorder if resuming

    batches = [reviews_ordered[i: i + batch_size] for i in range(0, len(reviews_ordered), batch_size)]
    remaining = batches[done:]

    if not remaining:
        if log_fn: log_fn("ok", f"All {done} batches already extracted.")
        return 0

    clients = get_clients()
    n_keys = len(clients)
    concurrency = min(MAX_PARALLEL, len(remaining), n_keys * 5)

    intel = load_intelligence(pid)
    demographic_hint = (intel or {}).get("demographic_profile", {}).get("summary", "general consumer")
    category_hint = f"{load_project(pid)['demographic_label']} products: {demographic_hint[:200]}"

    if log_fn:
        log_fn("info", f"Extracting {len(remaining)} batches (resuming from {done})...")
        log_fn("info", f"  {n_keys} API key(s), {concurrency} parallel workers, {EXTRACTION_TIMEOUT_MINUTES}min timeout")

    def _process_batch(batch_idx: int, batch: list[dict]) -> tuple[int, dict]:
        client = clients[batch_idx % n_keys]

        # --- Source-aware weighting ---
        # Amazon reviews are product-specific purchase feedback (high signal density).
        # Reddit comments are discussion-based (broader context but noisier, less
        # purchase-focused). We apply a 0.7x discount to Reddit's effective accuracy
        # so Amazon reviews dominate the signal when both sources are present.
        REDDIT_DISCOUNT = 0.7  # Reddit is broader discussion, not purchase-focused reviews

        weighted_batch = []
        for r in batch:
            acc = float(r.get("product_accuracy", 0.5))
            # Discount Reddit: their "accuracy" is thread_relevance * comment_score,
            # but even high-scoring Reddit comments are less purchase-specific than Amazon
            if r.get("source_type") == "reddit":
                acc *= REDDIT_DISCOUNT
            if acc < 0.2:
                continue  # skip off-demographic noise
            weighted_batch.append(r)
            # 2x weight for high-confidence Amazon reviews (verified purchase feedback)
            if acc >= 0.85:
                weighted_batch.append(r)

        signals = extract_signals_batch(weighted_batch, category_hint, client)
        # Track source composition per batch for downstream analysis.
        # These counts feed into cluster-level source mix reporting and
        # single-source warnings during clustering.
        signals["_source_weights"] = {
            "amazon_high_acc": sum(1 for r in batch if r.get("source_type") != "reddit" and float(r.get("product_accuracy", 0)) >= 0.9),
            "amazon_med_acc": sum(1 for r in batch if r.get("source_type") != "reddit" and 0.7 <= float(r.get("product_accuracy", 0)) < 0.9),
            "reddit": sum(1 for r in batch if r.get("source_type") == "reddit"),
            "csv": sum(1 for r in batch if r.get("source_type") == "csv"),
        }
        return (batch_idx, signals)

    # Run batches in parallel, collect results in order
    results: dict[int, dict] = {}
    deadline = time.time() + EXTRACTION_TIMEOUT_MINUTES * 60

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_process_batch, i, batch): i
            for i, batch in enumerate(remaining)
        }

        for future in as_completed(futures):
            if time.time() > deadline:
                if log_fn: log_fn("warn", f"⏱ {EXTRACTION_TIMEOUT_MINUTES}min timeout — proceeding with {len(results)}/{len(remaining)} batches")
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            try:
                batch_idx, signals = future.result()
                results[batch_idx] = signals
                cur = done + len(results)
                if log_fn:
                    pp = len(signals.get("pain_points", []))
                    pt = len(signals.get("purchase_triggers", []))
                    db = len(signals.get("deal_breakers", []))
                    log_fn("data", f"  Batch {cur}/{len(batches)} → {pp} pain pts, {pt} triggers, {db} deal breakers")
            except Exception as e:
                batch_idx = futures[future]
                if log_fn: log_fn("warn", f"  Batch {done + batch_idx + 1} failed: {e}")

    # Write results in order (only contiguous from start to avoid gaps)
    new_done = 0
    with open(sp, "a") as out:
        for i in range(len(remaining)):
            if i not in results:
                break
            out.write(json.dumps(results[i]) + "\n")
            out.flush()
            new_done += 1

    p = load_project(pid)
    p["stages"]["signals"] = "complete"
    save_project(p)
    if log_fn: log_fn("ok", f"Extraction done — {done + new_done} total batches ({new_done} new)")
    return new_done

# ---------------------------------------------------------------------------
# Pipeline: Clustering
# ---------------------------------------------------------------------------
# This is where RAW DATA becomes STATISTICS. Two key operations:
#
# 1. _count_with_confidence: For each signal text (pain point, trigger, etc.),
#    counts how often it appears across batches and computes 95% bootstrap
#    confidence intervals. This is the statistical backbone — downstream
#    decision_weights and emotional_profile are derived from these frequencies,
#    NOT from LLM-invented numbers.
#
# 2. run_clustering: Embeds signal batches using SentenceTransformer, then
#    runs KMeans with auto-optimized k (silhouette + intra/inter metrics).
#    Stability analysis (Adjusted Rand Index across 10 seeds) tells us how
#    reliable the cluster boundaries are. Each cluster gets:
#    - Frequency-based signal rankings with bootstrap CIs
#    - Source mix tracking (Amazon vs Reddit vs CSV)
#    - Friction tolerance distribution (feeds into emotional_profile)
#    - Single-source warnings (clusters dominated by one platform)

def _count_with_confidence(items: list[str], n_batches: int, n_bootstrap: int = 1000) -> list[dict]:
    """Count signal frequency and compute bootstrap 95% confidence intervals.

    Returns list of {signal, count, frequency, ci_lower, ci_upper, cross_source}.
    Only signals appearing in >= 2 batches are included (filters noise).
    """
    if not items or n_batches == 0:
        return []

    counts = Counter(items)

    # Filter: must appear in >= 2 batches to be considered a real signal
    counts = Counter({k: v for k, v in counts.items() if v >= 2})
    if not counts:
        # Fallback: just return top items without CI
        counts = Counter(items)

    results = []
    total = len(items)
    for signal, count in counts.most_common(15):
        freq = count / total if total else 0

        # Bootstrap CI: resample and compute frequency variance
        boot_freqs = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(items, size=len(items), replace=True)
            boot_count = sum(1 for s in sample if s == signal)
            boot_freqs.append(boot_count / len(sample))

        ci_lower = float(np.percentile(boot_freqs, 2.5))
        ci_upper = float(np.percentile(boot_freqs, 97.5))

        results.append({
            "signal": signal,
            "count": count,
            "frequency": round(freq, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
        })

    return results


def _cluster_stability_score(embeddings: np.ndarray, k: int, n_runs: int = 10) -> float:
    """Run KMeans n_runs times with different seeds and measure label agreement.

    Returns Adjusted Rand Index averaged across pairs — 1.0 = perfectly stable,
    0.0 = random assignment each time.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    all_labels = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        all_labels.append(km.fit_predict(embeddings))

    # Pairwise ARI across all runs
    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    return float(np.mean(ari_scores))


def _signal_text(item) -> str:
    """Extract text from a signal item that may be a plain string or a dict with CI annotations."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("signal", item.get("outcome", item.get("trigger", str(item))))
    return str(item)


def _signal_texts(items: list) -> list[str]:
    """Extract text strings from a list of signal items (handles both old str and new dict formats)."""
    return [_signal_text(item) for item in items]


def run_clustering(pid: str, n_override: int | None, sweep_min: int, sweep_max: int, log_fn=None, random_state: int = 42) -> dict:
    from sklearn.cluster import KMeans
    from pipeline.cluster import embed_signals, score_clustering, combined_score

    sp = signals_path(pid)
    if not sp.exists():
        raise ValueError("No signals file — run extraction first.")

    signals_list = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
    if len(signals_list) < 3:
        raise ValueError(f"Only {len(signals_list)} signal batches — need 3+.")

    if log_fn: log_fn("info", f"Embedding {len(signals_list)} signal batches...")
    embeddings = embed_signals(signals_list)

    weights = {"silhouette": 0.5, "intra_similarity": 0.3, "inter_distance": 0.2}
    sweep_max = min(sweep_max, len(signals_list) - 1)

    if n_override:
        best_k = n_override
        sweep_results: list[dict] = []
    else:
        if log_fn: log_fn("info", f"Sweeping k={sweep_min}..{sweep_max} for optimal clusters...")
        sweep_results = []
        best_k, best_score = sweep_min, -1.0
        for k in range(sweep_min, sweep_max + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(embeddings)
            metrics = score_clustering(embeddings, labels)
            cs = combined_score(metrics, weights)
            sweep_results.append({"k": k, **metrics, "combined_score": cs})
            if cs > best_score:
                best_score = cs
                best_k = k
        if log_fn: log_fn("ok", f"Best k={best_k} (score={best_score:.3f})")

    # Stability analysis: run KMeans with multiple seeds to verify cluster boundaries
    if log_fn: log_fn("info", f"Running stability analysis (10 seeds)...")
    stability = _cluster_stability_score(embeddings, best_k, n_runs=10)
    stability_label = "high" if stability >= 0.8 else "moderate" if stability >= 0.5 else "low"
    if log_fn: log_fn("data", f"  Cluster stability (ARI): {stability:.3f} ({stability_label})")
    if stability < 0.5 and log_fn:
        log_fn("warn", f"  Low stability — cluster boundaries shift across runs. Consider re-running with a different seed.")

    # Log cluster weight variance across seeds
    if log_fn and best_k >= 2:
        weight_runs = []
        for seed in range(5):
            km_test = KMeans(n_clusters=best_k, random_state=seed, n_init=5)
            test_labels = km_test.fit_predict(embeddings)
            sizes = np.bincount(test_labels, minlength=best_k)
            weight_runs.append(sizes / sizes.sum())
        weight_std = np.std(np.array(weight_runs), axis=0)
        log_fn("data", f"  Cluster weight std across seeds: {', '.join(f'{s:.3f}' for s in weight_std)}")

    if log_fn: log_fn("info", f"Building {best_k} clusters (seed={random_state})...")
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    metrics = score_clustering(embeddings, labels)

    # Track which sources contributed to each batch
    source_per_batch = []
    for s in signals_list:
        sw = s.get("_source_weights", {})
        source_per_batch.append(sw)

    result: dict[str, Any] = {
        "chosen_n_clusters": best_k,
        "sweep_results": sweep_results,
        "quality_metrics": {
            **metrics,
            "combined_score": combined_score(metrics, weights),
            "stability_ari": round(stability, 4),
            "stability_label": stability_label,
        },
    }

    for c in range(best_k):
        mask = labels == c
        indices = [i for i, m in enumerate(mask) if m]
        centroid = embeddings[mask].mean(axis=0)
        dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        rep_idx = [indices[i] for i in np.argsort(dists)[:5].tolist()]

        all_pain, all_outcomes, all_breakers = [], [], []
        friction_counts: dict[str, int] = {}
        trigger_list = []

        # Track source mix for this cluster
        cluster_sources = {"amazon": 0, "reddit": 0, "csv": 0}
        for idx in indices:
            s = signals_list[idx]
            all_pain.extend(
                sig for pp in s.get("pain_points", [])
                if isinstance(pp, dict)
                and (sig := (pp.get("signal") or pp.get("pain_point") or pp.get("text") or "").strip())
            )
            all_outcomes.extend(
                out for do in s.get("desired_outcomes", [])
                if isinstance(do, dict)
                and (out := (do.get("outcome") or do.get("desired_outcome") or do.get("text") or "").strip())
            )
            all_breakers.extend(db for db in s.get("deal_breakers", []) if isinstance(db, str))
            trigger_list.extend(t.get("trigger", "") for t in s.get("purchase_triggers", []) if isinstance(t, dict))
            ft = s.get("friction_tolerance", "medium")
            friction_counts[ft] = friction_counts.get(ft, 0) + 1

            sw = s.get("_source_weights", {})
            cluster_sources["amazon"] += sw.get("amazon_high_acc", 0) + sw.get("amazon_med_acc", 0)
            cluster_sources["reddit"] += sw.get("reddit", 0)
            cluster_sources["csv"] += sw.get("csv", 0)

        # Compute signal frequencies with bootstrap confidence intervals
        n_batches = len(indices)
        pain_stats = _count_with_confidence(all_pain, n_batches)
        outcome_stats = _count_with_confidence(all_outcomes, n_batches)
        trigger_stats = _count_with_confidence(trigger_list, n_batches)

        # Cross-source validation: flag signals that only appear from one source type
        total_source_reviews = sum(cluster_sources.values())
        source_mix = {k: round(v / total_source_reviews, 3) if total_source_reviews else 0 for k, v in cluster_sources.items()}
        single_source = sum(1 for v in source_mix.values() if v > 0.9)

        # Compute deal breaker frequencies with CIs (same rigor as pain points)
        breaker_stats = _count_with_confidence(all_breakers, n_batches)

        # Store full friction distribution for data-driven emotional profile
        # Maps: {"low": N, "medium": M, "high": H} — derived from LLM extraction
        friction_total = sum(friction_counts.values()) or 1

        result[f"cluster_{c}"] = {
            "member_count": int(mask.sum()),
            "representative_batches": [signals_list[i] for i in rep_idx],
            "source_mix": source_mix,
            "single_source_warning": single_source > 0,
            "aggregate_signals": {
                "top_pain_points": pain_stats[:12],
                "top_desired_outcomes": outcome_stats[:12],
                "top_deal_breakers": breaker_stats[:10],
                "top_purchase_triggers": trigger_stats[:8],
                "dominant_friction_tolerance": max(friction_counts, key=friction_counts.get) if friction_counts else "medium",
                # Full friction distribution — feeds into emotional_profile computation
                "friction_tolerance_distribution": {
                    k: round(v / friction_total, 3) for k, v in friction_counts.items()
                },
            },
        }
        if log_fn:
            src_str = " / ".join(f"{k}:{v:.0%}" for k, v in source_mix.items() if v > 0)
            warn = " ⚠ single-source" if single_source > 0 else ""
            top_pain = pain_stats[0]["signal"][:50] if pain_stats else "n/a"
            ci = f" (CI: {pain_stats[0]['ci_95_lower']:.0%}–{pain_stats[0]['ci_95_upper']:.0%})" if pain_stats else ""
            log_fn("data", f"  Cluster {c}: {int(mask.sum())} batches [{src_str}] — top: {top_pain}{ci}{warn}")

    # --- Validation ---
    cluster_keys = [k for k in result if k.startswith("cluster_")]
    total_members = sum(result[k]["member_count"] for k in cluster_keys)
    if total_members > 0:
        weight_sum = sum(result[k]["member_count"] / total_members for k in cluster_keys)
        if abs(weight_sum - 1.0) > 0.01:
            if log_fn: log_fn("warn", f"Cluster weights sum to {weight_sum:.4f} (expected ~1.0)")
        elif log_fn:
            log_fn("ok", f"Cluster weights validated (sum={weight_sum:.4f})")

    for cid in cluster_keys:
        agg = result[cid].get("aggregate_signals", {})
        missing = [f for f in ("top_pain_points", "top_desired_outcomes", "top_deal_breakers",
                               "top_purchase_triggers", "dominant_friction_tolerance")
                   if f not in agg]
        if missing:
            if log_fn: log_fn("warn", f"{cid}: missing fields: {', '.join(missing)}")
            for f in missing:
                agg[f] = [] if f != "dominant_friction_tolerance" else "medium"

    clusters_path(pid).write_text(json.dumps(result, indent=2))
    p = load_project(pid)
    p["stages"]["clusters"] = "complete"
    save_project(p)
    return result

# ---------------------------------------------------------------------------
# Pipeline: Persona synthesis (rich)
# ---------------------------------------------------------------------------
# The synthesis step combines DATA-DRIVEN computation with LLM interpretation.
# Clear separation of responsibilities:
#
# DATA-DRIVEN (computed before LLM call, enforced after):
#   - decision_weights: from signal frequency counts, cluster-size-weighted
#   - emotional_profile: baseline_patience from friction tolerance distribution,
#     trust_starting_point from trigger/breaker ratio, frustration_decay from
#     pain point density
#   - deal_breakers: validated against cluster data post-hoc
#
# LLM-SYNTHESIZED (requires human-like interpretation):
#   - behavioral_rules: translates data signals into testable agent rules
#   - voice_sample: creates authentic inner monologue using data vocabulary
#   - trigger_map: maps cluster signals to conversion/abandonment conditions
#   - purchase_journey: narrative interpretation of the behavioral data
#
# Post-validation enforces data values even if the LLM ignores instructions.
# The _provenance field in the output tracks what's data vs LLM for transparency.

_SYNTHESIS_PROMPT = """\
You are the world's best consumer behavioral researcher building a synthetic AI persona.

DEMOGRAPHIC: {label}
CONTEXT: {description}
DATA: {n_reviews} reviews from {n_sources} sources | {n_batches} signal batches | Quality: {quality}

CLUSTER TRAITS (sorted by frequency — these are the real behavioral patterns from the data):
{traits_json}

DATA QUALITY METRICS:
{quality_json}

━━━ PRE-COMPUTED DATA-DRIVEN METRICS (use these, do NOT invent your own) ━━━

The following values were computed statistically from the actual cluster data.
You MUST use these exact values — do not override them with your own estimates.

{precomputed_json}

IMPORTANT:
- "decision_weights_from_data" — these are the real signal frequencies from the data.
  You must use these as your decision_weights. You may relabel the factor names to be
  cleaner/shorter (e.g. "Battery dies too fast" → "battery_life") but the WEIGHTS must
  match the pre-computed values. Normalize so they sum to 1.0.
- "emotional_profile_from_data" — use these exact values for emotional_profile.
- "vocabulary_from_data" — phrases_they_use MUST be drawn from this list.

━━━ YOUR TASK ━━━

Build the most accurate, deeply representative persona possible. An AI agent will literally
embody this persona while browsing websites — every behavioral rule must be SPECIFIC and TESTABLE.

The output must be grounded entirely in the cluster data. No generic filler.
Every behavioral rule must trace back to a specific pain point, trigger, or deal breaker
from the cluster data. If you can't cite which cluster signal a rule comes from, don't include it.

Return valid JSON (no markdown fences) with ALL of these fields:

{{
  "id": "{safe_id}",
  "label": "{label}",
  "segment": {{
    "role": "<specific life stage>",
    "tech_savviness": "<low|medium|high — infer from data signals about tech frustration>",
    "price_sensitivity": "<low|medium|high|very_high — infer from price-related pain points frequency>",
    "risk_tolerance": "<low|medium|high — infer from deal breaker count and trust data>",
    "context": "<2 sentences — their core situation and what shapes their buying>"
  }},
  "goals": [
    {{"goal": "<specific thing they want to accomplish on a website>", "priority": <1-5>}}
  ],
  "constraints": ["<specific constraint from the data — cite which pain point or deal breaker>"],
  "decision_weights": "<USE the pre-computed decision_weights_from_data — relabel factors but keep weights>",
  "behavioral_rules": [
    "<specific testable rule grounded in a cluster signal: 'Immediately exits if pricing page requires signup before showing prices' (from pain point: hidden pricing)>"
  ],
  "emotional_profile": "<USE the pre-computed emotional_profile_from_data exactly>",
  "deal_breakers": ["<concrete binary condition — must appear in cluster top_deal_breakers>"],
  "voice_sample": "<3-4 paragraph inner monologue using vocabulary_from_data phrases>",
  "browsing_patterns": {{
    "typical_session_length_minutes": <int>,
    "pages_before_decision": <int>,
    "tab_behavior": "<how they use browser tabs>",
    "primary_device": "<device + context>",
    "time_of_day": "<when they typically research/buy>"
  }},
  "purchase_journey": {{
    "discovery": "<how they first find products/services — specific behavior from data>",
    "evaluation": "<what they look at, in what order — grounded in trigger/pain data>",
    "decision_moment": "<the specific threshold — grounded in purchase triggers from data>",
    "post_purchase_expectation": "<what they expect — grounded in desired outcomes from data>"
  }},
  "trigger_map": {{
    "converts_when": ["<from top_purchase_triggers in cluster data>", ...],
    "abandons_when": ["<from top_pain_points in cluster data>", ...],
    "trusts_when": ["<from top_desired_outcomes in cluster data>", ...],
    "becomes_suspicious_when": ["<from top_deal_breakers in cluster data>", ...]
  }},
  "vocabulary": {{
    "phrases_they_use": ["<MUST be from vocabulary_from_data list>", ...],
    "resonates_with": ["<marketing language that maps to desired_outcomes in data>", ...],
    "turned_off_by": ["<language that maps to pain_points in data>", ...]
  }},
  "inner_monologue_samples": [
    "<Thought at pricing page — reference specific price sensitivity from data>",
    "<Thought when they see social proof — reference trust signals from data>",
    "<Thought when navigation is confusing — reference friction tolerance from data>",
    "<Thought at signup form — reference specific deal breakers from data>",
    "<Thought when they find what they're looking for — reference desired outcomes from data>"
  ],
  "anti_patterns": [
    "<Common mistake — each must cite which data signal it contradicts>"
  ]
}}

Rules:
- decision_weights: USE the pre-computed values from decision_weights_from_data, just clean up the labels
- emotional_profile: USE the pre-computed values from emotional_profile_from_data exactly
- Minimum: 5 goals, 4 constraints, 10 behavioral_rules, 5 deal_breakers
- behavioral_rules must use action verbs and be specific enough to drive agent behavior
- All voice sample vocabulary must come from vocabulary_from_data
- trigger_map: minimum 4 items per key, each must trace to a specific cluster signal
- inner_monologue_samples: must feel authentic and reference specific data points
- Every claim must be traceable to the cluster data — no generic filler
"""

def run_persona_synthesis(pid: str, quality: dict, log_fn=None) -> None:
    from pipeline.export import extract_trait, build_rag_index

    cp = clusters_path(pid)
    if not cp.exists():
        raise ValueError("No clusters file — run clustering first.")

    clusters = json.loads(cp.read_text())
    intel = load_intelligence(pid) or {}
    p = load_project(pid)

    client = get_client()
    cluster_keys = [k for k in clusters if k.startswith("cluster_")]

    # Label each cluster as a behavioral trait
    if log_fn: log_fn("info", f"Labeling {len(cluster_keys)} behavioral trait clusters...")
    traits: list[dict] = []
    total_members = sum(clusters[k]["member_count"] for k in cluster_keys)

    for i, cid in enumerate(cluster_keys):
        if log_fn: log_fn("info", f"  Labeling cluster {i+1}/{len(cluster_keys)}...")
        try:
            trait = extract_trait(cid, clusters[cid], client)
        except Exception as e:
            agg = clusters[cid].get("aggregate_signals", {})
            top = _signal_texts(agg.get("top_pain_points", []))
            trait = {
                "label": f"Trait {i+1}",
                "description": f"Behavioral cluster around: {top[0][:80] if top else 'unknown pattern'}",
                "key_phrases": top[:3],
                "tone": "pragmatic",
                "cluster_id": cid,
            }
        count = clusters[cid]["member_count"]
        trait["cluster_id"] = cid
        trait["frequency"] = round(count / total_members, 3) if total_members else 0
        trait["member_count"] = count
        traits.append(trait)
        if log_fn: log_fn("data", f"    [{trait['frequency']:.0%}] {trait['label']} — {trait.get('tone','')}")

    traits.sort(key=lambda t: -t.get("frequency", 0))

    # =====================================================================
    # DATA-DRIVEN PRE-COMPUTATION
    # Instead of letting the LLM hallucinate numbers, we compute key metrics
    # directly from the cluster data and pass them as constraints.
    # =====================================================================
    if log_fn: log_fn("info", "Computing data-driven metrics from cluster signals...")

    # --- 1. Decision weights from signal frequencies ---
    # Count how often each decision factor appears across ALL clusters,
    # weighted by cluster size. Factors are derived from actual pain points,
    # triggers, and outcomes — not invented by the LLM.
    factor_counts: Counter = Counter()
    total_signals = 0
    for cid in cluster_keys:
        agg = clusters[cid].get("aggregate_signals", {})
        weight = clusters[cid]["member_count"]
        for pp in agg.get("top_pain_points", []):
            signal = pp["signal"] if isinstance(pp, dict) else pp
            # Map pain points to decision factors by extracting the theme
            factor_counts[signal] += (pp.get("count", 1) if isinstance(pp, dict) else 1) * weight
            total_signals += (pp.get("count", 1) if isinstance(pp, dict) else 1) * weight
        for pt in agg.get("top_purchase_triggers", []):
            trigger = pt["signal"] if isinstance(pt, dict) else (pt.get("trigger", pt) if isinstance(pt, dict) else pt)
            factor_counts[trigger] += (pt.get("count", 1) if isinstance(pt, dict) else 1) * weight
            total_signals += (pt.get("count", 1) if isinstance(pt, dict) else 1) * weight

    # Normalize to get frequency-based weights (top 8 factors)
    top_factors = factor_counts.most_common(8)
    factor_total = sum(c for _, c in top_factors) or 1
    data_decision_weights = {
        signal: round(count / factor_total, 3) for signal, count in top_factors
    }
    if log_fn:
        for f, w in list(data_decision_weights.items())[:5]:
            log_fn("data", f"  Decision factor: {f[:50]} = {w:.1%}")

    # --- 2. Emotional profile from friction tolerance distribution ---
    # Aggregate friction tolerance across all clusters, weighted by size.
    # "low" friction tolerance → low patience, "high" → high patience.
    friction_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
    weighted_patience = 0.0
    total_friction_weight = 0
    for cid in cluster_keys:
        agg = clusters[cid].get("aggregate_signals", {})
        dist = agg.get("friction_tolerance_distribution", {})
        cluster_weight = clusters[cid]["member_count"]
        for level, pct in dist.items():
            weighted_patience += friction_map.get(level, 0.5) * pct * cluster_weight
            total_friction_weight += pct * cluster_weight
    baseline_patience = round(weighted_patience / total_friction_weight, 3) if total_friction_weight else 0.5

    # Trust starting point: derived from ratio of positive triggers to deal breakers.
    # More deal breakers relative to triggers = lower starting trust (skeptical).
    total_triggers = sum(
        len(clusters[cid].get("aggregate_signals", {}).get("top_purchase_triggers", []))
        for cid in cluster_keys
    )
    total_breakers = sum(
        len(clusters[cid].get("aggregate_signals", {}).get("top_deal_breakers", []))
        for cid in cluster_keys
    )
    # Sigmoid-like mapping: equal triggers/breakers → 0.5, more breakers → lower trust
    trust_ratio = total_triggers / max(total_triggers + total_breakers, 1)
    trust_starting_point = round(0.2 + 0.6 * trust_ratio, 3)  # range [0.2, 0.8]

    # Frustration decay: how fast patience drops. More pain points = faster decay.
    avg_pain_per_cluster = sum(
        len(clusters[cid].get("aggregate_signals", {}).get("top_pain_points", []))
        for cid in cluster_keys
    ) / max(len(cluster_keys), 1)
    # Normalize: 0-3 pain points → low decay (0.1), 10+ → high decay (0.4)
    frustration_decay = round(min(0.4, max(0.1, avg_pain_per_cluster * 0.04)), 3)

    data_emotional_profile = {
        "baseline_patience": baseline_patience,
        "trust_starting_point": trust_starting_point,
        "frustration_decay": frustration_decay,
    }
    if log_fn:
        log_fn("data", f"  Emotional profile: patience={baseline_patience}, trust={trust_starting_point}, decay={frustration_decay}")

    # --- 3. Collect all unique vocabulary from cluster data ---
    all_phrases = set()
    for cid in cluster_keys:
        for batch in clusters[cid].get("representative_batches", []):
            for pp in batch.get("pain_points", []):
                if isinstance(pp, dict):
                    all_phrases.add(pp.get("signal", ""))
            for do in batch.get("desired_outcomes", []):
                if isinstance(do, dict):
                    all_phrases.add(do.get("outcome", ""))
    all_phrases.discard("")

    # --- 4. Build pre-computed data block for the prompt ---
    precomputed = {
        "decision_weights_from_data": data_decision_weights,
        "emotional_profile_from_data": data_emotional_profile,
        "cluster_stability": clusters.get("quality_metrics", {}).get("stability_label", "unknown"),
        "cluster_stability_ari": clusters.get("quality_metrics", {}).get("stability_ari", 0),
        "vocabulary_from_data": sorted(all_phrases)[:50],
    }

    if log_fn: log_fn("info", "Synthesizing full persona from all traits (this is the big call)...")
    safe_id = p["demographic_label"].lower().replace(" ", "_").replace("-", "_")

    reviews = load_all_reviews(pid)
    n_sources = len(set(r.get("source_type", "unknown") for r in reviews))

    prompt_filled = _SYNTHESIS_PROMPT.format(
        label=p["demographic_label"],
        description=p["demographic_description"],
        n_reviews=quality["total"],
        n_sources=n_sources,
        n_batches=quality["batches_estimate"],
        quality=quality["score"].upper(),
        traits_json=json.dumps(traits, indent=2),
        quality_json=json.dumps(quality, indent=2),
        safe_id=safe_id,
        precomputed_json=json.dumps(precomputed, indent=2),
    )

    raw = llm([{"role": "user", "content": prompt_filled}], max_tokens=6000, model=MODEL_SMART)
    persona = _parse_llm_json(raw)

    persona["traits"] = [
        {
            "label": t["label"],
            "description": t.get("description", ""),
            "key_phrases": t.get("key_phrases", []),
            "frequency": t.get("frequency", 0.0),
            "tone": t.get("tone", "unknown"),
            "cluster_id": t["cluster_id"],
        }
        for t in traits
    ]
    persona["data_quality"] = quality
    persona["demographic_description"] = p["demographic_description"]
    persona["generated_at"] = datetime.utcnow().isoformat()

    # =====================================================================
    # POST-VALIDATION: Enforce data-driven values the LLM may have ignored
    # =====================================================================
    if log_fn: log_fn("info", "Validating persona against data-driven metrics...")

    # 1. Force emotional_profile to match pre-computed values
    persona["emotional_profile"] = data_emotional_profile
    if log_fn: log_fn("ok", f"  emotional_profile enforced: {data_emotional_profile}")

    # 2. Validate decision_weights sum to 1.0; if LLM botched it, use pre-computed
    dw = persona.get("decision_weights", {})
    if isinstance(dw, dict) and dw:
        dw_sum = sum(dw.values())
        if abs(dw_sum - 1.0) > 0.05:
            if log_fn: log_fn("warn", f"  decision_weights sum={dw_sum:.3f}, re-normalizing")
            dw = {k: round(v / dw_sum, 3) for k, v in dw.items()}
            # Fix rounding so it sums to exactly 1.0
            diff = round(1.0 - sum(dw.values()), 3)
            if dw:
                first_key = next(iter(dw))
                dw[first_key] = round(dw[first_key] + diff, 3)
            persona["decision_weights"] = dw
            if log_fn: log_fn("ok", f"  decision_weights re-normalized to sum=1.0")
    else:
        # LLM didn't return valid weights — use pre-computed directly
        if log_fn: log_fn("warn", f"  decision_weights missing/invalid, using data-derived values")
        persona["decision_weights"] = data_decision_weights

    # 3. Validate deal_breakers appear in cluster data
    cluster_breakers = set()
    for cid in cluster_keys:
        for db in clusters[cid].get("aggregate_signals", {}).get("top_deal_breakers", []):
            if isinstance(db, dict):
                cluster_breakers.add(db.get("signal", "").lower())
            elif isinstance(db, str):
                cluster_breakers.add(db.lower())
    persona_breakers = persona.get("deal_breakers", [])
    if persona_breakers and cluster_breakers:
        # Check how many are grounded vs invented
        grounded = sum(1 for b in persona_breakers if any(
            cb in b.lower() or b.lower() in cb for cb in cluster_breakers
        ))
        grounding_pct = grounded / len(persona_breakers) if persona_breakers else 0
        if log_fn:
            log_fn("data", f"  deal_breakers: {grounded}/{len(persona_breakers)} grounded in cluster data ({grounding_pct:.0%})")
        if grounding_pct < 0.5:
            if log_fn: log_fn("warn", f"  Low grounding — most deal breakers may be LLM-invented")

    # 4. Stamp provenance metadata so downstream consumers know what's data vs LLM
    persona["_provenance"] = {
        "emotional_profile": "data-driven (friction tolerance distribution + trigger/breaker ratio)",
        "decision_weights": "data-driven (signal frequency counts, cluster-weighted)",
        "behavioral_rules": "LLM-synthesized from cluster signals",
        "voice_sample": "LLM-synthesized using data vocabulary",
        "deal_breakers": f"{grounding_pct:.0%} grounded in cluster data" if persona_breakers and cluster_breakers else "unknown",
        "cluster_stability": clusters.get("quality_metrics", {}).get("stability_label", "unknown"),
    }

    od = outputs_dir(pid)
    od.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(pid)

    # Save persona.json
    (od / f"{stem}_persona.json").write_text(json.dumps(persona, indent=2))
    if log_fn: log_fn("ok", f"{stem}_persona.json written")

    # Generate rich persona.md
    if log_fn: log_fn("info", "Generating rich markdown context document...")
    md_content = _generate_rich_markdown(persona, traits, intel, quality, p)
    (od / f"{stem}_persona.md").write_text(md_content)
    if log_fn: log_fn("ok", f"{stem}_persona.md written")

    # Build RAG index
    rag_entries = build_rag_index(traits, clusters)
    # Enrich with inner monologue samples
    for sample in persona.get("inner_monologue_samples", []):
        rag_entries.append({
            "review_id": f"monologue_{uuid.uuid4().hex[:6]}",
            "trait_label": "Inner Monologue",
            "text": sample,
            "tone": "authentic_voice",
        })
    with open(od / f"{stem}_rag_index.jsonl", "w") as f:
        for entry in rag_entries:
            f.write(json.dumps(entry) + "\n")
    if log_fn: log_fn("ok", f"{stem}_rag_index.jsonl written ({len(rag_entries)} entries)")

    p["stages"]["persona"] = "complete"
    save_project(p)

def _generate_rich_markdown(persona: dict, traits: list[dict], intel: dict, quality: dict, proj: dict) -> str:
    seg = persona.get("segment", {})
    ep = persona.get("emotional_profile", {})
    pj = persona.get("purchase_journey", {})
    tm = persona.get("trigger_map", {})
    vocab = persona.get("vocabulary", {})
    mono = persona.get("inner_monologue_samples", [])
    anti = persona.get("anti_patterns", [])
    dw = persona.get("decision_weights", {})

    lines = [
        f"# {persona.get('label', 'Persona')} — Synthetic Persona Context Document",
        f"",
        f"> Generated from **{quality['total']:,} reviews** across **{quality['n_sources']} sources** | Data quality: **{quality['score'].upper()}** | {persona.get('generated_at', '')[:10]}",
        f"",
        f"---",
        f"",
        f"## Who Is This Person",
        f"",
        f"**Role:** {seg.get('role', 'N/A')}  ",
        f"**Context:** {seg.get('context', 'N/A')}  ",
        f"**Tech savviness:** {seg.get('tech_savviness', 'N/A')} | **Price sensitivity:** {seg.get('price_sensitivity', 'N/A')} | **Risk tolerance:** {seg.get('risk_tolerance', 'N/A')}",
        f"",
        persona.get("demographic_description", ""),
        f"",
        f"---",
        f"",
        f"## Behavioral Trait Breakdown",
        f"",
        f"These traits were derived from clustering {quality['batches_estimate']} signal batches:",
        f"",
    ]

    for t in sorted(traits, key=lambda x: -x.get("frequency", 0)):
        lines += [
            f"### {t['label']} ({t.get('frequency', 0):.0%} of signal)",
            f"",
            f"{t.get('description', '')}",
            f"",
            f"**Tone:** {t.get('tone', 'N/A')}  ",
            f"**Key phrases from data:**",
        ]
        for phrase in t.get("key_phrases", []):
            lines.append(f'> "{phrase}"')
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## How They Make Decisions",
        f"",
        f"### What Drives Them (Decision Weights)",
        f"",
    ]
    for factor, weight in sorted(dw.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
        lines.append(f"- **{factor.replace('_',' ').title()}** `{bar}` {weight:.0%}")
    lines.append("")

    lines += [
        f"### Their Purchase Journey",
        f"",
        f"**Discovery:** {pj.get('discovery', 'N/A')}",
        f"",
        f"**Evaluation:** {pj.get('evaluation', 'N/A')}",
        f"",
        f"**Decision moment:** {pj.get('decision_moment', 'N/A')}",
        f"",
        f"**Post-purchase expectation:** {pj.get('post_purchase_expectation', 'N/A')}",
        f"",
        f"---",
        f"",
        f"## Goals & Constraints",
        f"",
        f"### What They Want to Accomplish",
        f"",
    ]
    for g in sorted(persona.get("goals", []), key=lambda x: -x.get("priority", 0)):
        lines.append(f"- (Priority {g.get('priority','?')}) {g.get('goal', '')}")

    lines += [f"", f"### Their Constraints", f""]
    for c in persona.get("constraints", []):
        lines.append(f"- {c}")

    lines += [
        f"",
        f"---",
        f"",
        f"## Trigger Map",
        f"",
        f"| Trigger Type | Specific Conditions |",
        f"|---|---|",
        f"| **Converts when** | {' / '.join(tm.get('converts_when', [])[:3])} |",
        f"| **Abandons when** | {' / '.join(tm.get('abandons_when', [])[:3])} |",
        f"| **Trusts when** | {' / '.join(tm.get('trusts_when', [])[:3])} |",
        f"| **Suspicious when** | {' / '.join(tm.get('becomes_suspicious_when', [])[:3])} |",
        f"",
        f"### Deal Breakers (Immediate Exit)",
        f"",
    ]
    for db in persona.get("deal_breakers", []):
        lines.append(f"- 🚫 {db}")

    lines += [
        f"",
        f"---",
        f"",
        f"## Behavioral Rules (for AI Agents)",
        f"",
        f"These rules govern exactly how this persona behaves when browsing:",
        f"",
    ]
    for i, rule in enumerate(persona.get("behavioral_rules", []), 1):
        lines.append(f"{i}. {rule}")

    lines += [
        f"",
        f"## Emotional Profile",
        f"",
        f"- **Baseline patience:** {ep.get('baseline_patience', 0.5):.0%} — {'high tolerance for friction' if ep.get('baseline_patience', 0.5) > 0.6 else 'low tolerance, exits quickly when frustrated'}",
        f"- **Starting trust:** {ep.get('trust_starting_point', 0.5):.0%} — {'starts somewhat trusting' if ep.get('trust_starting_point', 0.5) > 0.5 else 'starts skeptical, must be earned'}",
        f"- **Frustration decay:** {ep.get('frustration_decay', 0.3):.0%} per friction event",
        f"",
        f"**Browsing Patterns:**",
        f"",
    ]
    bp = persona.get("browsing_patterns", {})
    for k, v in bp.items():
        lines.append(f"- **{k.replace('_',' ').title()}:** {v}")

    lines += [
        f"",
        f"---",
        f"",
        f"## Language Guide",
        f"",
        f"### Phrases They Actually Use",
        f"",
    ]
    for phrase in vocab.get("phrases_they_use", []):
        lines.append(f'- "{phrase}"')

    lines += [f"", f"### Marketing Language That Resonates", f""]
    for phrase in vocab.get("resonates_with", []):
        lines.append(f"- ✓ {phrase}")

    lines += [f"", f"### Language That Turns Them Off", f""]
    for phrase in vocab.get("turned_off_by", []):
        lines.append(f"- ✗ {phrase}")

    lines += [
        f"",
        f"---",
        f"",
        f"## Inner Monologue Samples",
        f"",
        f"How this persona thinks while browsing — use these to calibrate agent responses:",
        f"",
    ]
    for sample in mono:
        lines.append(f"> {sample}")
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## Voice Sample",
        f"",
        f"*In their own words:*",
        f"",
        f"> {persona.get('voice_sample', '').replace(chr(10), chr(10) + '> ')}",
        f"",
        f"---",
        f"",
        f"## Anti-Patterns (What NOT to Do)",
        f"",
        f"Common mistakes when designing experiences for this demographic:",
        f"",
    ]
    for ap in anti:
        lines.append(f"- ⚠️ {ap}")

    lines += [
        f"",
        f"---",
        f"",
        f"## Data Quality Report",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total reviews | {quality['total']:,} |",
        f"| Sources | {', '.join(quality['source_breakdown'].keys())} |",
        f"| Avg review length | {quality['avg_length']} chars |",
        f"| Avg rating | {quality['avg_rating']}/5 |",
        f"| Negative reviews | {quality['low_rating_pct']:.0%} |",
        f"| Positive reviews | {quality['high_rating_pct']:.0%} |",
        f"| Signal batches | {quality['batches_estimate']} |",
        f"| Quality score | **{quality['score'].upper()}** |",
        f"",
    ]
    if quality["issues"]:
        lines.append("**Notes:**")
        for issue in quality["issues"]:
            lines.append(f"- ⚠️ {issue}")

    return "\n".join(lines)

# ===========================================================================
# Streamlit UI — Clean Wizard Layout
# (entire block skipped when PROXI_WORKER_MODE=1)
# ===========================================================================

if not _WORKER_MODE:
    # ── everything below is UI-only ──────────────────────────────────────────

    # --- Top bar ---
    col_logo, col_proj, col_key = st.columns([2, 5, 2])

    with col_logo:
       st.markdown('<div style="padding-top:0.4rem"><span style="font-size:1.1rem;font-weight:800;color:#e2e8f0">🧠 Proxi</span><span style="font-size:0.75rem;color:#475569;margin-left:6px">Training</span></div>', unsafe_allow_html=True)

    with col_key:
       _api_keys = _parse_keys()
       has_key = len(_api_keys) > 0
       if has_key:
           key_label = f"{len(_api_keys)} key{'s' if len(_api_keys) > 1 else ''}"
           st.markdown(f'<p style="text-align:right;color:#4ade80;font-size:0.78rem;margin-top:0.7rem;margin-bottom:0">● {key_label} loaded</p>', unsafe_allow_html=True)
       else:
           st.markdown('<p style="text-align:right;color:#f87171;font-size:0.78rem;margin-top:0.7rem;margin-bottom:0">● no API key</p>', unsafe_allow_html=True)

    # Project selector
    projects = list_projects()

    with col_proj:
       options = [""] + [p["id"] for p in projects]
       labels = ["— create new project —"] + [
           f"{p['name']}  —  {p.get('demographic_label', '')}  ({p.get('review_count', count_reviews(p['id']))} reviews)"
           for p in projects
       ]
       _default_idx = st.session_state.pop("_selected_idx", None)
       idx = st.selectbox("Project", options=range(len(options)), format_func=lambda i: labels[i], label_visibility="collapsed", index=_default_idx if _default_idx is not None else 0)
       selected_pid = options[idx]

    st.divider()

    # ===========================================================================
    # QUEUE — batch persona generation
    # ===========================================================================
    from data.queue_manager import (
        add_job, read_queue, remove_job, clear_finished, requeue_failed, requeue_all,
        is_worker_alive, start_worker, stop_worker, worker_pid,
        read_log, update_job, STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
    )

    def _archive_job_outputs(job: dict) -> str | None:
        """Copy existing output files to an archive subfolder. Returns archive path or None."""
        pid  = job.get("persona_id") or f"queue_{job['id']}"
        od   = outputs_dir(pid)
        stem = _output_stem(pid)
        files_to_archive = [
            od / f"{stem}_persona.json",
            od / f"{stem}_persona.md",
            od / f"{stem}_rag_index.jsonl",
        ]
        existing = [f for f in files_to_archive if f.exists()]
        if not existing:
            return None
        import shutil
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = od / "archive" / ts
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in existing:
            shutil.copy2(f, archive_dir / f.name)
        return str(archive_dir)

    # ── Queue header ──────────────────────────────────────────────────────────
    _q_jobs_all = read_queue()
    _q_pending  = sum(1 for j in _q_jobs_all if j["status"] == STATUS_PENDING)
    _q_running  = sum(1 for j in _q_jobs_all if j["status"] == STATUS_RUNNING)
    _q_done     = sum(1 for j in _q_jobs_all if j["status"] == STATUS_DONE)
    _q_failed   = sum(1 for j in _q_jobs_all if j["status"] == STATUS_FAILED)
    _q_total    = len(_q_jobs_all)

    _alive = is_worker_alive()
    _wpid  = worker_pid()

    # Header row: title + worker status + controls
    qh1, qh2, qh3, qh4 = st.columns([3, 2, 1, 1])
    with qh1:
        if _alive:
            st.markdown(
                f'<div style="padding-top:0.5rem">'
                f'<span style="font-size:1rem;font-weight:700;color:#e2e8f0">⚡ Batch Queue</span>'
                f'  <span class="pulse" style="color:#4ade80;font-size:0.8rem">● Worker running</span>'
                f'  <span style="color:#475569;font-size:0.75rem">(PID {_wpid})</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding-top:0.5rem">'
                '<span style="font-size:1rem;font-weight:700;color:#e2e8f0">⚡ Batch Queue</span>'
                '  <span style="color:#f87171;font-size:0.8rem">● Worker stopped</span></div>',
                unsafe_allow_html=True,
            )
    with qh2:
        st.markdown(
            f'<div class="queue-stats">'
            f'<div class="qstat"><div class="qstat-num" style="color:#94a3b8">{_q_pending}</div><div class="qstat-lbl">Pending</div></div>'
            f'<div class="qstat"><div class="qstat-num" style="color:#facc15">{_q_running}</div><div class="qstat-lbl">Running</div></div>'
            f'<div class="qstat"><div class="qstat-num" style="color:#4ade80">{_q_done}</div><div class="qstat-lbl">Done</div></div>'
            f'<div class="qstat"><div class="qstat-num" style="color:#f87171">{_q_failed}</div><div class="qstat-lbl">Failed</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with qh3:
        if st.button("▶ Start" if not _alive else "▶ Running", disabled=_alive, use_container_width=True):
            new_pid = start_worker()
            st.success(f"Worker started (PID {new_pid})")
            st.rerun()
    with qh4:
        if st.button("■ Stop", disabled=not _alive, use_container_width=True):
            stop_worker()
            st.warning("Worker stopped")
            st.rerun()

    st.caption("Tip: `caffeinate -i python worker.py` to prevent macOS sleep during overnight runs.")

    st.divider()

    with st.expander("➕ Add Personas to Queue", expanded=False):

        if "queue_rows" not in st.session_state:
            st.session_state.queue_rows = [{"project_name": "", "label": "", "description": ""}]

        with st.form("queue_add_form", clear_on_submit=False):
            col_add, col_cfg = st.columns([3, 1])

            with col_cfg:
                q_target  = st.number_input("Target reviews", min_value=2000, max_value=10000, value=5000, step=500)
                q_threads = st.number_input("Reddit threads/sub", min_value=1, max_value=20, value=8)
                q_comments = st.number_input("Comments/thread", min_value=50, max_value=300, value=150)

            with col_add:
                for i, row in enumerate(st.session_state.queue_rows):
                    rc1, rc2, rc3 = st.columns([1, 1, 2])
                    with rc1:
                        st.session_state.queue_rows[i]["project_name"] = st.text_input(
                            "Company", value=row.get("project_name", ""),
                            placeholder="Disgo",
                            key=f"qpname_{i}", label_visibility="collapsed" if i > 0 else "visible"
                        )
                    with rc2:
                        st.session_state.queue_rows[i]["label"] = st.text_input(
                            "Demographic title", value=row["label"],
                            placeholder="Consumer App User",
                            key=f"qlabel_{i}", label_visibility="collapsed" if i > 0 else "visible"
                        )
                    with rc3:
                        st.session_state.queue_rows[i]["description"] = st.text_area(
                            "Description", value=row["description"],
                            placeholder="A user of a consumer app focused on…",
                            key=f"qdesc_{i}", height=68, label_visibility="collapsed" if i > 0 else "visible"
                        )

            fc1, fc2, fc3 = st.columns([1, 1, 2])
            with fc1:
                add_row = st.form_submit_button("＋ Add Row")
            with fc2:
                submit_queue = st.form_submit_button("Add All to Queue ▶", type="primary")

        if add_row:
            st.session_state.queue_rows.append({"label": "", "description": ""})
            st.rerun()

        if submit_queue:
            cfg = {"target_total": q_target, "max_threads": q_threads, "max_comments": q_comments}
            added = 0
            for row in st.session_state.queue_rows:
                pname = row.get("project_name", "").strip()
                lbl   = row["label"].strip()
                desc  = row["description"].strip()
                if lbl and desc:
                    add_job(lbl, desc, project_name=pname or lbl, config=cfg)
                    added += 1
            if added:
                st.session_state.queue_rows = [{"label": "", "description": ""}]
                st.success(f"Added {added} job{'s' if added != 1 else ''} to queue")
                st.rerun()
            else:
                st.error("Fill in at least one label + description")

    # ── Queue status table (outside the form expander) ──────────────────────
    _HEARTBEAT_FILE = Path("projects/worker_heartbeat.txt")

    _STEP_META = {
        "create_project": (5,  "Setting up project"),
        "intelligence":   (13, "Generating intelligence"),
        "amazon":         (25, "Pulling Amazon reviews"),
        "reddit":         (38, "Scraping Reddit"),
        "hackernews":     (48, "Scraping Hacker News"),
        "appstore":       (55, "Scraping App Store"),
        "playstore":      (62, "Scraping Google Play Store"),
        "review_gate":    (66, "Validating review count"),
        "extraction":     (78, "Extracting signals"),
        "clustering":     (90, "Clustering"),
        "synthesis":      (96, "Synthesising persona"),
        "done":           (100, "Complete"),
    }

    def _get_heartbeat() -> dict:
        try:
            d = {}
            for ln in _HEARTBEAT_FILE.read_text().splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    d[k.strip()] = v.strip()
            return d
        except Exception:
            return {}

    jobs = read_queue()
    if not jobs:
        st.markdown('<div style="color:#475569;font-size:0.85rem;padding:0.5rem 0 1rem">Queue is empty — open "Add Personas to Queue" above to get started.</div>', unsafe_allow_html=True)
    else:
        # ── Bulk actions ──────────────────────────────────────────────────
        ba1, ba2, ba3, _ = st.columns([1, 1, 1, 3])
        with ba1:
            if st.button("↺ Rerun All", use_container_width=True, help="Reset all done + failed jobs to pending with target=8,000 reviews. Existing reviews kept — only new data added."):
                for j in [j for j in jobs if j["status"] == STATUS_DONE]:
                    _archive_job_outputs(j)
                n = requeue_all(target_total=8_000)
                st.success(f"Queued {n} job(s) for rerun (target: 8,000 reviews). Old outputs archived.")
                st.rerun()
        with ba2:
            if st.button("↻ Retry Failed", use_container_width=True):
                n = requeue_failed()
                st.info(f"Reset {n} failed job(s) to pending")
                st.rerun()
        with ba3:
            if st.button("✕ Clear Done", use_container_width=True):
                clear_finished()
                st.rerun()

        heartbeat = _get_heartbeat()
        hb_job_id = heartbeat.get("job", "")
        hb_step   = heartbeat.get("step", "")

        # ── Job cards ─────────────────────────────────────────────────────
        for job in jobs:
            s     = job["status"]
            pname = job.get("project_name", job["label"])
            lbl   = job["label"]
            dur   = f"{int(job['duration_s'])//60}m {int(job['duration_s'])%60}s" if job.get("duration_s") else ""

            _badge_cls = {"pending":"s-pending","running":"s-running","done":"s-done","failed":"s-failed"}.get(s,"s-pending")
            _card_cls  = {"pending":"qcard-pending","running":"qcard-running","done":"qcard-done","failed":"qcard-failed"}.get(s,"qcard-pending")

            st.markdown(f'<div class="qcard {_card_cls}">', unsafe_allow_html=True)

            jc1, jc2 = st.columns([7, 1])
            with jc1:
                st.markdown(
                    f'<div class="qcard-title">{pname} <span style="color:#374151;font-weight:400">·</span> {lbl}</div>'
                    f'<div class="qcard-meta">'
                    f'<span class="status-badge {_badge_cls}">{s}</span>'
                    + (f'  <span style="color:#475569">{dur}</span>' if dur else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )
            with jc2:
                if s == STATUS_PENDING:
                    if st.button("✕", key=f"rm_{job['id']}", use_container_width=True, help="Remove from queue"):
                        remove_job(job["id"]); st.rerun()
                elif s in (STATUS_DONE, STATUS_FAILED):
                    if st.button("↺", key=f"rerun_{job['id']}", use_container_width=True, help="Rerun with target=8,000 reviews (existing reviews kept, old outputs archived)"):
                        if s == STATUS_DONE:
                            archived = _archive_job_outputs(job)
                            if archived:
                                st.toast(f"Archived to …/archive/{Path(archived).name}")
                        _new_cfg = dict(job.get("config") or {})
                        _new_cfg["target_total"] = 8_000
                        update_job(job["id"], status=STATUS_PENDING, error=None, duration_s=None, config=_new_cfg)
                        st.rerun()

            # Running: live progress + log
            if s == STATUS_RUNNING:
                step_now = hb_step if hb_job_id == job["id"] else ""
                pct, step_label = _STEP_META.get(step_now, (2, "Starting…"))
                st.progress(pct / 100, text=f"{step_label}  —  {pct}%")
                log_text = read_log(job["id"], tail=12)
                if log_text:
                    st.code(log_text, language=None)

            # Done: compact output downloads
            elif s == STATUS_DONE:
                _pid  = job.get("persona_id") or f"queue_{job['id']}"
                _od   = outputs_dir(_pid)
                _stem = _output_stem(_pid)
                fn_json = f"{_stem}_persona.json"
                fn_md   = f"{_stem}_persona.md"
                fn_rag  = f"{_stem}_rag_index.jsonl"
                jp, mp, rp = _od / fn_json, _od / fn_md, _od / fn_rag
                with st.expander("📄 Outputs", expanded=False):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        if jp.exists():
                            st.download_button("⬇ JSON", jp.read_bytes(), fn_json, "application/json", key=f"dl_j_{job['id']}", use_container_width=True)
                        else:
                            st.caption("json missing")
                    with dc2:
                        if mp.exists():
                            st.download_button("⬇ Markdown", mp.read_bytes(), fn_md, "text/markdown", key=f"dl_m_{job['id']}", use_container_width=True)
                        else:
                            st.caption("md missing")
                    with dc3:
                        if rp.exists():
                            st.download_button("⬇ RAG Index", rp.read_bytes(), fn_rag, "application/x-ndjson", key=f"dl_r_{job['id']}", use_container_width=True)
                        else:
                            st.caption("rag missing")

            # Failed: error + log
            elif s == STATUS_FAILED:
                if job.get("error"):
                    st.markdown(f'<div style="color:#f87171;font-size:0.8rem;margin-top:4px">{job["error"][:200]}</div>', unsafe_allow_html=True)
                with st.expander("View log"):
                    st.code(read_log(job["id"], tail=40) or "(empty)", language=None)

            st.markdown("</div>", unsafe_allow_html=True)

        # Auto-refresh every 3s while a job is running
        if any(j["status"] == STATUS_RUNNING for j in jobs):
            time.sleep(3)
            st.rerun()

    # ===========================================================================
    # NEW PROJECT — setup wizard
    # ===========================================================================
    if not selected_pid:
       st.markdown("## Start: Define Your Demographic")
       st.markdown("The more specific your description, the more accurately DeepSeek can identify targeted products and subreddits.")

       with st.form("create_project"):
           col1, col2 = st.columns([1, 2])
           with col1:
               pname = st.text_input("Project name", placeholder="College Students 2024")
               plabel = st.text_input("Demographic label", placeholder="College Student",
                                      help="Short label used as the persona ID")
           with col2:
               pdesc = st.text_area(
                   "Describe this demographic in detail",
                   height=120,
                   placeholder=(
                       "US college students aged 18–24, living on or off campus. "
                       "Extremely price-sensitive — most have under $200/month discretionary budget. "
                       "Shop primarily on mobile late at night. Heavily influenced by peer recommendations "
                       "and TikTok. Distrust corporate marketing. Value authenticity and transparency. "
                       "Tech-savvy but impatient with slow or confusing UX."
                   ),
                   help="Include: age range, income/budget, shopping behavior, values, frustrations, tech habits"
               )
           submitted = st.form_submit_button("🚀 Create Project & Generate Intelligence", type="primary", use_container_width=True)

       if submitted:
           if not pname or not plabel or not pdesc:
               st.error("Fill in all three fields.")
           elif not has_key:
               st.error("Set OPENROUTER_API_KEY first.")
           else:
               proj = new_project(pname, plabel, pdesc)
               st.session_state["active_pid"] = proj["id"]
               st.session_state["log"] = []

               log_ph = st.empty()

               def _log(kind, msg):
                   color = {"info": "#60a5fa", "ok": "#4ade80", "warn": "#fbbf24", "data": "#a78bfa"}.get(kind, "#ffffff")
                   st.session_state["log"].append(f'<span style="color:{color}">{msg}</span>')
                   log_ph.markdown(
                       '<div class="log-box">' + "<br>".join(st.session_state["log"][-30:]) + "</div>",
                       unsafe_allow_html=True,
                   )

               with st.spinner("Asking DeepSeek to map your demographic to specific products and subreddits..."):
                   try:
                       generate_intelligence(proj["id"], pdesc, log_fn=_log)
                       st.success("Intelligence generated! Reloading...")
                       time.sleep(1)
                       st.rerun()
                   except Exception as e:
                       st.error(f"Failed: {e}")
                       st.exception(e)
       st.stop()

    # ===========================================================================
    # EXISTING PROJECT — main workspace
    # ===========================================================================

    # Sync active project
    if "active_pid" in st.session_state and st.session_state["active_pid"] == selected_pid:
       pass
    else:
       st.session_state["active_pid"] = selected_pid
       st.session_state["log"] = []

    proj = load_project(selected_pid)
    if not proj:
       st.error("Project not found.")
       st.stop()

    # Backfill missing keys from old project format
    if "review_count" not in proj:
       proj["review_count"] = count_reviews(selected_pid)
    if "review_sources" not in proj:
       proj["review_sources"] = []
    if "stages" not in proj:
       proj["stages"] = {"intelligence":"pending","reviews":"empty","signals":"pending","clusters":"pending","persona":"pending"}
    # Backfill stage keys
    for stage_key in ("intelligence", "reviews", "signals", "clusters", "persona"):
       if stage_key not in proj["stages"]:
           proj["stages"][stage_key] = "pending"
    save_project(proj)

    intel = load_intelligence(selected_pid)
    stages = proj["stages"]

    # Step status bar
    def step_html(label, status):
       cls = "step-done" if status == "complete" else "step-active" if status in ("ready","running") else "step-pending"
       icon = "✓" if status == "complete" else "●" if status in ("ready","running") else "○"
       return f'<div class="step {cls}">{icon} {label}</div>'

    _rsrcs = proj.get("review_sources", [])
    _has_reddit  = any(s["label"].startswith("Reddit") for s in _rsrcs)
    _has_hn      = any(s["label"].startswith(("HackerNews", "HN", "Hacker")) for s in _rsrcs)
    _has_appstore  = any(s["label"].startswith("App Store") for s in _rsrcs)
    _has_playstore = any(s["label"].startswith("Play Store") for s in _rsrcs)
    st.markdown(
       f'<div class="step-row">'
       f'{step_html("1 Intel", stages["intelligence"])}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("2 Amazon", "complete" if stages["reviews"]=="ready" and proj["review_count"]>0 else stages["reviews"])}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("3 Reddit", "complete" if _has_reddit else "pending")}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("4 HN", "complete" if _has_hn else "pending")}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("5 App Store", "complete" if _has_appstore else "pending")}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("6 Play Store", "complete" if _has_playstore else "pending")}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("7 Extract", stages["signals"])}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("8 Cluster", stages["clusters"])}'
       f'<span class="step-arrow">›</span>'
       f'{step_html("9 Persona", stages["persona"])}'
       f'</div>',
       unsafe_allow_html=True,
    )

    _back_col, _title_col = st.columns([1, 8])
    with _back_col:
        if st.button("← Home", use_container_width=True):
            st.query_params.clear()
            st.session_state["_selected_idx"] = 0
            st.rerun()
    with _title_col:
        st.markdown(f"## {proj['name']}")
        st.caption(f"**{proj['demographic_label']}** — {proj['demographic_description'][:180]}{'...' if len(proj['demographic_description'])>180 else ''}")
    st.divider()

    # ===========================================================================
    # SECTION 1: Intelligence (products + subreddits)
    # ===========================================================================
    with st.expander("📡 Step 1: Demographic Intelligence", expanded=(stages["intelligence"] != "complete")):
       if stages["intelligence"] != "complete" or not intel:
           st.warning("Intelligence not yet generated.")
           if st.button("Generate Intelligence", type="primary"):
               log_ph = st.empty()
               st.session_state["log"] = []
               def _log(k,m):
                   c={"info":"#60a5fa","ok":"#4ade80","warn":"#fbbf24","data":"#a78bfa"}.get(k,"#fff")
                   st.session_state["log"].append(f'<span style="color:{c}">{m}</span>')
                   log_ph.markdown('<div class="log-box">'+"<br>".join(st.session_state["log"][-20:])+"</div>",unsafe_allow_html=True)
               try:
                   intel = generate_intelligence(selected_pid, proj["demographic_description"], log_fn=_log)
                   st.rerun()
               except Exception as e:
                   st.error(f"{e}")
       else:
           products = intel.get("products", [])
           subreddits = intel.get("subreddits", [])
           dp = intel.get("demographic_profile", {})

           col1, col2 = st.columns([3, 2])
           with col1:
               st.markdown(f"**{len(products)} targeted products identified**")
               for prod in sorted(products, key=lambda p: -p.get("accuracy", 0)):
                   acc = prod.get("accuracy", 0)
                   acc_cls = "confidence-high" if acc >= 0.85 else "confidence-med" if acc >= 0.7 else "confidence-low"
                   kw_preview = ", ".join(prod.get("review_keywords", [])[:4])
                   st.markdown(
                       f'<div class="product-card {acc_cls}">'
                       f'<strong>{prod["name"]}</strong> <span style="color:#888;font-size:0.8rem">({prod.get("category","?").replace("raw_review_","").replace("_"," ")})</span><br>'
                       f'<span style="color:#4ade80;font-size:0.85rem">▲ {acc:.0%} demographic accuracy</span><br>'
                       f'<span style="color:#6a6d7a;font-size:0.78rem">Keywords: {kw_preview}</span><br>'
                       f'<span style="color:#5a5d6a;font-size:0.75rem">{prod.get("why","")}</span>'
                       f'</div>',
                       unsafe_allow_html=True,
                   )

           with col2:
               st.markdown(f"**{len(subreddits)} subreddits**")
               for sub in sorted(subreddits, key=lambda s: -s.get("relevance", 0)):
                   rel = sub.get("relevance", 0)
                   color = "#4ade80" if rel >= 0.85 else "#fbbf24" if rel >= 0.7 else "#f87171"
                   st.markdown(
                       f'<div class="product-card" style="border-left:3px solid {color}">'
                       f'<strong>r/{sub["name"]}</strong> <span style="color:{color};font-size:0.8rem">{rel:.0%}</span><br>'
                       f'<span style="color:#6a6d7a;font-size:0.78rem">{sub.get("data_value","")}</span>'
                       f'</div>',
                       unsafe_allow_html=True,
                   )

               st.markdown("**Demographic Keywords**")
               kw_html = " ".join(f'<span class="sub-chip">{k}</span>' for k in dp.get("core_keywords", []))
               st.markdown(kw_html, unsafe_allow_html=True)

    # ===========================================================================
    # SECTION 2: Amazon Reviews
    # ===========================================================================
    with st.expander("🛒 Step 2: Pull Amazon Reviews", expanded=(stages["reviews"] == "empty" and stages["intelligence"] == "complete")):
       if stages["intelligence"] != "complete":
           st.warning("Complete Step 1 first.")
       else:
           products = intel.get("products", []) if intel else []
           col1, col2 = st.columns([3, 1])

           with col1:
               all_prod_names = [p["name"] for p in sorted(products, key=lambda p: -p.get("accuracy", 0))]
               # Default: select top products with accuracy >= 0.8
               default_selected = [p["name"] for p in products if p.get("accuracy", 0) >= 0.80]
               selected_products = st.multiselect(
                   "Select products to pull reviews for",
                   options=all_prod_names,
                   default=default_selected[:12],
                   help="Higher accuracy = more demographically targeted reviews",
               )
           with col2:
               max_per_product = st.number_input("Max reviews/product", 50, 500, 200, 50)
               st.caption(f"~{len(selected_products) * max_per_product:,} reviews estimated")

           if proj["review_count"] > 0:
               st.success(f"{proj['review_count']:,} Amazon reviews already loaded. You can pull more.")

           if st.button("⬇️ Pull Amazon Reviews", type="primary", disabled=not selected_products):
               st.session_state["log"] = []
               log_ph = st.empty()

               def _log(k, m):
                   c = {"info":"#60a5fa","ok":"#4ade80","warn":"#fbbf24","data":"#a78bfa"}.get(k,"#fff")
                   st.session_state["log"].append(f'<span style="color:{c}">{m}</span>')
                   log_ph.markdown('<div class="log-box">' + "<br>".join(st.session_state["log"][-30:]) + "</div>", unsafe_allow_html=True)

               with st.status("Streaming from HuggingFace...", expanded=True) as status_box:
                   try:
                       added = pull_amazon_reviews(selected_pid, intel, selected_products, int(max_per_product), log_fn=_log)
                       proj = load_project(selected_pid)
                       status_box.update(label=f"Done — {added:,} reviews added", state="complete")
                       st.rerun()
                   except Exception as e:
                       status_box.update(label="Failed", state="error")
                       st.error(f"{e}")
                       st.exception(e)

    # ===========================================================================
    # SECTION 3: Reddit + CSV
    # ===========================================================================
    with st.expander("📋 Step 3: Add Reddit & Custom Reviews (Recommended)", expanded=False):
       st.markdown("Adding Reddit data significantly improves signal quality — it captures authentic unprompted opinions.")

       # Reddit — automated scraper
       st.markdown("### Reddit (Auto-Scraper)")
       if intel:
           subs = intel.get("subreddits", [])
           top_subs = sorted(subs, key=lambda s: -s.get("relevance", 0))[:10]
           sub_html = " ".join(f'<span class="sub-chip">r/{s["name"]} ({s.get("relevance",0):.0%})</span>' for s in top_subs)
           st.markdown(f"**Suggested subreddits:** {sub_html}", unsafe_allow_html=True)
       else:
           top_subs = []

       st.markdown(
           "Automatically scrapes top threads from selected subreddits. "
           "Deduplicates recurring weekly threads (keeps the best instance). "
           "Extracts all substantive comments via Reddit's JSON API."
       )

       # Subreddit selection
       available_subs = [s["name"] for s in top_subs] if top_subs else []
       custom_subs = st.text_input("Add subreddits (comma-separated)", placeholder="college, StudentLife, personalfinance", key="reddit_custom_subs")
       if custom_subs:
           for s in custom_subs.split(","):
               s = s.strip().replace("r/", "")
               if s and s not in available_subs:
                   available_subs.append(s)

       selected_subs = st.multiselect("Select subreddits to scrape", available_subs, default=available_subs[:6], key="reddit_selected_subs")

       col_r1, col_r2 = st.columns(2)
       max_threads = col_r1.slider("Threads per subreddit", 3, 20, 8, key="reddit_max_threads")
       max_comments = col_r2.slider("Comments per thread", 50, 500, 150, key="reddit_max_comments")

       if st.button("🔍 Scrape Reddit", type="primary", disabled=not selected_subs):
           reddit_log = st.empty()
           reddit_logs: list[str] = []

           def _reddit_log(kind, msg):
               color = {"info": "#60a5fa", "ok": "#4ade80", "warn": "#fbbf24", "data": "#a78bfa"}.get(kind, "#ffffff")
               prefix = {"info": "→", "ok": "✓", "warn": "⚠", "data": "  "}.get(kind, " ")
               reddit_logs.append(f'<span style="color:{color}">{prefix} {msg}</span>')
               reddit_log.markdown(
                   '<div class="log-box">' + "<br>".join(reddit_logs[-30:]) + "</div>",
                   unsafe_allow_html=True,
               )

           with st.status(f"Scraping {len(selected_subs)} subreddits...", expanded=True):
               try:
                   added = scrape_reddit_for_project(
                       selected_pid,
                       selected_subs,
                       max_threads_per_sub=max_threads,
                       max_comments_per_thread=max_comments,
                       log_fn=_reddit_log,
                   )
                   proj = load_project(selected_pid)
                   st.success(f"Done — {added:,} Reddit comments added. Total reviews: {proj['review_count']:,}")
                   st.rerun()
               except Exception as e:
                   st.error(f"Scraping failed: {e}")
                   st.exception(e)

       st.divider()

       # Hacker News — Algolia API (free, no key)
       st.markdown("### Hacker News (Ask HN + Tech Discussions)")
       st.markdown(
           "Searches HN comments via the free Algolia API — no key required. "
           "Great for surfacing authentic developer/founder opinions about tools and products."
       )

       if intel:
           products = intel.get("products", [])
           default_hn_queries = [p["name"] for p in products[:4] if p.get("accuracy", 0) >= 0.6]
           # Add demographic-flavoured queries
           kws = intel.get("demographic_profile", {}).get("core_keywords", [])
           if kws:
               default_hn_queries.append(" ".join(kws[:3]))
       else:
           default_hn_queries = []

       hn_queries_raw = st.text_area(
           "Search queries (one per line)",
           value="\n".join(default_hn_queries[:6]),
           height=120,
           key="hn_queries",
           help="Each query is searched independently. Use product names, problem descriptions, or technology terms.",
       )
       hn_max_per_query = st.slider("Max comments per query", 50, 500, 200, key="hn_max_per_query")

       if st.button("🟠 Scrape Hacker News", type="primary"):
           hn_queries = [q.strip() for q in hn_queries_raw.splitlines() if q.strip()]
           if not hn_queries:
               st.warning("Add at least one search query.")
           else:
               hn_log = st.empty()
               hn_logs: list[str] = []

               def _hn_log(kind, msg):
                   color = {"info": "#60a5fa", "ok": "#4ade80", "warn": "#fbbf24", "data": "#a78bfa"}.get(kind, "#ffffff")
                   prefix = {"info": "→", "ok": "✓", "warn": "⚠", "data": "  "}.get(kind, " ")
                   hn_logs.append(f'<span style="color:{color}">{prefix} {msg}</span>')
                   hn_log.markdown(
                       '<div class="log-box">' + "<br>".join(hn_logs[-30:]) + "</div>",
                       unsafe_allow_html=True,
                   )

               target_total = st.session_state.get("budget_target", 5000)
               with st.status(f"Searching {len(hn_queries)} HN queries…", expanded=True):
                   try:
                       added = scrape_hn_for_project(
                           selected_pid,
                           hn_queries,
                           max_per_query=hn_max_per_query,
                           target_total=target_total,
                           log_fn=_hn_log,
                       )
                       proj = load_project(selected_pid)
                       st.success(f"Done — {added:,} HN comments added. Total reviews: {proj['review_count']:,}")
                       st.rerun()
                   except Exception as e:
                       st.error(f"HN scrape failed: {e}")
                       st.exception(e)

       st.divider()

       # App Store — iTunes Search API + app-store-scraper
       st.markdown("### App Store (Apple)")
       st.markdown("Pulls reviews via the iTunes Search API. No key required.")

       if intel:
           default_appstore_queries = [p["name"] for p in intel.get("products", [])[:5] if p.get("accuracy", 0) >= 0.6]
       else:
           default_appstore_queries = []

       appstore_queries_raw = st.text_area(
           "App names to search (one per line)",
           value="\n".join(default_appstore_queries),
           height=80,
           key="appstore_queries_raw",
       )
       appstore_max = st.slider("Max reviews per app", 50, 500, 200, key="appstore_max")

       if st.button("🍎 Scrape App Store", type="primary"):
           appstore_queries = [q.strip() for q in appstore_queries_raw.splitlines() if q.strip()]
           if not appstore_queries:
               st.warning("Add at least one app name.")
           else:
               as_log = st.empty()
               as_logs: list[str] = []

               def _as_log(kind, msg):
                   color = {"info": "#60a5fa", "ok": "#4ade80", "warn": "#fbbf24", "data": "#a78bfa"}.get(kind, "#ffffff")
                   prefix = {"info": "→", "ok": "✓", "warn": "⚠", "data": "  "}.get(kind, " ")
                   as_logs.append(f'<span style="color:{color}">{prefix} {msg}</span>')
                   as_log.markdown(
                       '<div class="log-box">' + "<br>".join(as_logs[-30:]) + "</div>",
                       unsafe_allow_html=True,
                   )

               target_total = st.session_state.get("budget_target", 5000)
               with st.status(f"Searching {len(appstore_queries)} App Store queries…", expanded=True):
                   try:
                       added = scrape_appstore_for_project(
                           selected_pid,
                           appstore_queries,
                           max_per_app=appstore_max,
                           target_total=target_total,
                           log_fn=_as_log,
                       )
                       proj = load_project(selected_pid)
                       st.success(f"Done — {added:,} App Store reviews added. Total: {proj['review_count']:,}")
                       st.rerun()
                   except Exception as e:
                       st.error(f"App Store scrape failed: {e}")
                       st.exception(e)

       st.divider()

       # Google Play Store
       st.markdown("### Google Play Store")
       st.markdown("Searches Play Store and pulls reviews. No key required.")

       if intel:
           default_playstore_queries = [p["name"] for p in intel.get("products", [])[:5] if p.get("accuracy", 0) >= 0.6]
       else:
           default_playstore_queries = []

       playstore_queries_raw = st.text_area(
           "App names to search (one per line)",
           value="\n".join(default_playstore_queries),
           height=80,
           key="playstore_queries_raw",
       )
       playstore_max = st.slider("Max reviews per app", 50, 500, 200, key="playstore_max")

       if st.button("🤖 Scrape Play Store", type="primary"):
           playstore_queries = [q.strip() for q in playstore_queries_raw.splitlines() if q.strip()]
           if not playstore_queries:
               st.warning("Add at least one app name.")
           else:
               ps_log = st.empty()
               ps_logs: list[str] = []

               def _ps_log(kind, msg):
                   color = {"info": "#60a5fa", "ok": "#4ade80", "warn": "#fbbf24", "data": "#a78bfa"}.get(kind, "#ffffff")
                   prefix = {"info": "→", "ok": "✓", "warn": "⚠", "data": "  "}.get(kind, " ")
                   ps_logs.append(f'<span style="color:{color}">{prefix} {msg}</span>')
                   ps_log.markdown(
                       '<div class="log-box">' + "<br>".join(ps_logs[-30:]) + "</div>",
                       unsafe_allow_html=True,
                   )

               target_total = st.session_state.get("budget_target", 5000)
               with st.status(f"Searching {len(playstore_queries)} Play Store queries…", expanded=True):
                   try:
                       added = scrape_playstore_for_project(
                           selected_pid,
                           playstore_queries,
                           max_per_app=playstore_max,
                           target_total=target_total,
                           log_fn=_ps_log,
                       )
                       proj = load_project(selected_pid)
                       st.success(f"Done — {added:,} Play Store reviews added. Total: {proj['review_count']:,}")
                       st.rerun()
                   except Exception as e:
                       st.error(f"Play Store scrape failed: {e}")
                       st.exception(e)

       st.divider()

       # CSV
       st.markdown("### CSV Reviews (G2, Amazon, Trustpilot, custom)")
       st.markdown("Supports any CSV with a text/review column. Flexible column naming.")
       csv_files = st.file_uploader("Upload review CSVs", type=["csv"], accept_multiple_files=True, key="csv_upload")
       if csv_files:
           for cf in csv_files:
               reviews = parse_csv_reviews(cf.read(), source_label=cf.name)
               if not reviews:
                   st.warning(f"`{cf.name}` — no valid reviews found (need a text column, min 30 chars)")
               else:
                   with st.expander(f"Preview `{cf.name}` — {len(reviews)} reviews"):
                       for r in reviews[:3]:
                           st.markdown(f"**{'⭐'*int(r.get('rating',3))} ({r.get('rating','?')})** — {r.get('product','')}")
                           st.caption((r.get("text", ""))[:200])
                   col1, col2 = st.columns([4, 1])
                   col1.info(f"`{cf.name}` — {len(reviews)} valid reviews ready")
                   if col2.button("Add", key=f"add_csv_{cf.name}"):
                       total = append_reviews(selected_pid, reviews, source_label=cf.name)
                       st.success(f"Added {len(reviews)} reviews. Total: {total:,}")
                       st.rerun()

    # ===========================================================================
    # SECTION 4: Data Quality Check
    # ===========================================================================
    reviews_all = load_all_reviews(selected_pid)
    quality = analyze_quality(reviews_all)

    # Budget target slider — controls the 40%-per-source cap across all scrapers
    from data.budget import budget_summary, MIN_REVIEWS, MAX_REVIEWS, DEFAULT_TOTAL
    if "budget_target" not in st.session_state:
       st.session_state["budget_target"] = DEFAULT_TOTAL
    budget_target = st.slider(
       "Review target (controls 40%-per-source cap)",
       min_value=MIN_REVIEWS,
       max_value=MAX_REVIEWS,
       value=st.session_state["budget_target"],
       step=500,
       key="budget_target_slider",
       help="Each source may contribute at most 40% of this number. Minimum to build a persona: 2,000.",
    )
    st.session_state["budget_target"] = budget_target

    bsummary = budget_summary(reviews_all, budget_target)
    _src_colors = {"amazon": "#f59e0b", "reddit": "#ef4444", "hackernews": "#f97316", "csv": "#8b5cf6"}
    _bar_parts = []
    for src, info in bsummary["per_source"].items():
       color = _src_colors.get(src, "#6b7280")
       pct_of_target = info["count"] / budget_target if budget_target else 0
       cap_pct = 0.40
       capped_indicator = " ⚠ at cap" if info["at_cap"] else ""
       _title = f"{src}: {info['count']:,} ({info['pct']:.0%}){capped_indicator}"
       _bar_parts.append(
           f'<span style="display:inline-block;background:{color};width:{pct_of_target*100:.1f}%;'
           f'height:18px;vertical-align:middle;" title="{_title}"></span>'
       )
    progress_pct = min(1.0, quality["total"] / budget_target) if budget_target else 0
    remaining_pct = max(0.0, 1.0 - progress_pct)
    _bar_parts.append(
       f'<span style="display:inline-block;background:#1f2937;width:{remaining_pct*100:.1f}%;'
       f'height:18px;vertical-align:middle;border:1px dashed #374151;" title="remaining"></span>'
    )
    st.markdown(
       f'<div style="margin-bottom:4px"><strong>Source budget:</strong> {quality["total"]:,} / {budget_target:,} '
       f'({progress_pct:.0%}) — 40% cap per source ({int(budget_target*0.4):,} max each)</div>'
       f'<div style="width:100%;background:#111;border-radius:4px;overflow:hidden">{"".join(_bar_parts)}</div>'
       + "".join(
           f'<span style="font-size:0.75rem;color:{_src_colors.get(src,"#6b7280")};margin-right:12px">'
           f'■ {src} {info["count"]:,} ({info["pct"]:.0%})'
           + (" ⚠" if info["at_cap"] else "") + "</span>"
           for src, info in bsummary["per_source"].items()
       ),
       unsafe_allow_html=True,
    )
    if not bsummary["meets_minimum"]:
       st.warning(f"Need at least {MIN_REVIEWS:,} reviews to build a reliable persona. Currently at {quality['total']:,}.")

    with st.expander(f"📊 Data Quality — {quality['total']:,} reviews | {quality['score'].upper()}", expanded=True):
       badge_cls = f"badge-{quality['score']}"
       q_cols = st.columns(5)
       q_cols[0].metric("Total Reviews", f"{quality['total']:,}")
       q_cols[1].metric("Sources", quality["n_sources"])
       q_cols[2].metric("Avg Length", f"{quality['avg_length']} chars")
       q_cols[3].metric("Products", quality["n_products"])
       q_cols[4].metric("Quality", quality["score"].upper())

       if quality["issues"]:
           for issue in quality["issues"]:
               st.warning(issue)
       else:
           st.success("Data looks good! Sufficient volume, sources, and rating distribution.")

       # Source breakdown
       if quality["source_breakdown"]:
           col1, col2 = st.columns(2)
           with col1:
               st.markdown("**Source breakdown:**")
               for src, cnt in quality["source_breakdown"].items():
                   pct = cnt / quality["total"] if quality["total"] else 0
                   cap_mark = " ⚠ at cap" if bsummary["per_source"].get(src, {}).get("at_cap") else ""
                   st.markdown(f"- `{src}`: {cnt:,} ({pct:.0%}){cap_mark}")
           with col2:
               st.markdown("**Rating distribution (sentiment-inferred for Reddit/HN):**")
               for stars in range(5, 0, -1):
                   cnt = quality["rating_distribution"].get(stars, 0)
                   pct = cnt / quality["total"] if quality["total"] else 0
                   bar = "█" * int(pct * 20)
                   st.markdown(f"- {'⭐'*stars}: {cnt} `{bar}` {pct:.0%}")

    # ===========================================================================
    # SECTION 5+6: Run Pipeline
    # ===========================================================================
    st.divider()
    st.markdown("## Run Training Pipeline")

    if quality["total"] < 100:
       st.error("Need at least 100 reviews to run the pipeline. Add more data above.")
    elif quality["total"] < 300:
       st.warning(f"Only {quality['total']} reviews — results will be weak. Recommended: 300+. Continue anyway?")

    col1, col2, col3, col4 = st.columns(4)
    batch_size = col1.select_slider("Batch size", [15, 20, 30, 40, 50], value=30, help="Reviews per LLM extraction call")
    force_rerun = col2.checkbox("Re-run extraction from scratch", value=False)
    n_clusters_override = col3.number_input("Force cluster count (0 = auto)", 0, 20, 0)
    cluster_seed = col4.number_input("Cluster seed", 0, 999, 42, help="Change seed to test clustering stability")

    # Show pipeline stage status
    status_cols = st.columns(3)
    status_cols[0].markdown(f"**Extract:** {stages['signals']}")
    status_cols[1].markdown(f"**Cluster:** {stages['clusters']}")
    status_cols[2].markdown(f"**Persona:** {stages['persona']}")

    # Live log area
    log_ph = st.empty()

    if "log" not in st.session_state:
       st.session_state["log"] = []

    def _log(kind, msg):
       color = {"info": "#60a5fa", "ok": "#4ade80", "warn": "#fbbf24", "data": "#a78bfa"}.get(kind, "#ffffff")
       prefix = {"info": "→", "ok": "✓", "warn": "⚠", "data": "  "}.get(kind, " ")
       st.session_state["log"].append(f'<span style="color:{color}">{prefix} {msg}</span>')
       log_ph.markdown(
           '<div class="log-box">' + "<br>".join(st.session_state["log"][-40:]) + "</div>",
           unsafe_allow_html=True,
       )

    col_pipe1, col_pipe2 = st.columns([1, 1])

    with col_pipe1:
       run_full = st.button("▶ Run Full Pipeline (Extract → Cluster → Build Persona)", type="primary", use_container_width=True, disabled=quality["total"] < 50)
    with col_pipe2:
       run_everything = st.button("🚀 Run EVERYTHING (Pull Data → Extract → Cluster → Build)", use_container_width=True, disabled=(stages["intelligence"] != "complete"))

    if run_full or run_everything:
       if not has_key:
           st.error("Set OPENROUTER_API_KEY.")
       else:
           st.session_state["log"] = []

           with st.status("Running pipeline...", expanded=True) as status_box:

               # === AUTO-PULL DATA (only for "Run Everything") ===
               if run_everything and intel:
                   _auto_target = st.session_state.get("budget_target", DEFAULT_TOTAL)
                   products = intel.get("products", [])

                   # Amazon
                   _log("info", "━━━ AUTO: Pulling Amazon Reviews ━━━")
                   default_products = [p["name"] for p in products if p.get("accuracy", 0) >= 0.75]
                   if default_products:
                       try:
                           added = pull_amazon_reviews(selected_pid, intel, default_products, 200, log_fn=_log)
                           _log("ok", f"Amazon: {added:,} reviews added")
                           proj = load_project(selected_pid)
                       except Exception as e:
                           _log("warn", f"Amazon pull failed: {e} — continuing")

                   # Reddit
                   _log("info", "━━━ AUTO: Scraping Reddit ━━━")
                   subs_list = intel.get("subreddits", [])
                   sub_names = [s["name"] for s in sorted(subs_list, key=lambda s: -s.get("relevance", 0))[:6]]
                   if sub_names:
                       try:
                           added = scrape_reddit_for_project(
                               selected_pid, sub_names,
                               max_threads_per_sub=8, max_comments_per_thread=150,
                               target_total=_auto_target,
                               log_fn=_log,
                           )
                           _log("ok", f"Reddit: {added:,} comments added")
                           proj = load_project(selected_pid)
                       except Exception as e:
                           _log("warn", f"Reddit scrape failed: {e} — continuing")

                   # Hacker News
                   _log("info", "━━━ AUTO: Scraping Hacker News ━━━")
                   hn_queries = [p["name"] for p in products[:4] if p.get("accuracy", 0) >= 0.6]
                   kws = intel.get("demographic_profile", {}).get("core_keywords", [])
                   if kws:
                       hn_queries.append(" ".join(kws[:3]))
                   if hn_queries:
                       try:
                           added = scrape_hn_for_project(
                               selected_pid, hn_queries,
                               max_per_query=200,
                               target_total=_auto_target,
                               log_fn=_log,
                           )
                           _log("ok", f"HN: {added:,} comments added")
                           proj = load_project(selected_pid)
                       except Exception as e:
                           _log("warn", f"HN scrape failed: {e} — continuing")

                   # App Store
                   _log("info", "━━━ AUTO: Scraping App Store ━━━")
                   as_queries = [p["name"] for p in products[:5] if p.get("accuracy", 0) >= 0.6]
                   if as_queries:
                       try:
                           added = scrape_appstore_for_project(
                               selected_pid, as_queries,
                               max_per_app=200,
                               target_total=_auto_target,
                               log_fn=_log,
                           )
                           _log("ok", f"App Store: {added:,} reviews added")
                           proj = load_project(selected_pid)
                       except Exception as e:
                           _log("warn", f"App Store scrape failed: {e} — continuing")

                   # Google Play Store
                   _log("info", "━━━ AUTO: Scraping Google Play Store ━━━")
                   ps_queries = [p["name"] for p in products[:5] if p.get("accuracy", 0) >= 0.6]
                   if ps_queries:
                       try:
                           added = scrape_playstore_for_project(
                               selected_pid, ps_queries,
                               max_per_app=200,
                               target_total=_auto_target,
                               log_fn=_log,
                           )
                           _log("ok", f"Play Store: {added:,} reviews added")
                           proj = load_project(selected_pid)
                       except Exception as e:
                           _log("warn", f"Play Store scrape failed: {e} — continuing")

                   # Reload quality after all data pulls
                   reviews_all = load_all_reviews(selected_pid)
                   quality = analyze_quality(reviews_all)

               _log("info", f"Starting pipeline on {quality['total']:,} reviews...")

               # Stage 1: Extract
               try:
                   _log("info", "━━━ STAGE 1: Signal Extraction ━━━")
                   new_b = run_extraction(selected_pid, int(batch_size), force_rerun, log_fn=_log)
                   _log("ok", f"Extraction complete — {new_b} new batches processed")
                   proj = load_project(selected_pid)
               except Exception as e:
                   status_box.update(label="Extraction failed", state="error")
                   st.error(f"Extraction failed: {e}")
                   st.stop()

               # Stage 2: Cluster
               try:
                   _log("info", "━━━ STAGE 2: Clustering ━━━")
                   n_sig = sum(1 for _ in signals_path(selected_pid).read_text().splitlines() if _.strip())
                   sweep_max = min(10, max(3, n_sig // 5))
                   result = run_clustering(
                       selected_pid,
                       n_override=int(n_clusters_override) if n_clusters_override > 0 else None,
                       sweep_min=3,
                       sweep_max=sweep_max,
                       log_fn=_log,
                       random_state=int(cluster_seed),
                   )
                   _log("ok", f"Clustering complete — {result['chosen_n_clusters']} clusters")
                   proj = load_project(selected_pid)
               except Exception as e:
                   status_box.update(label="Clustering failed", state="error")
                   st.error(f"Clustering failed: {e}")
                   st.stop()

               # Stage 3: Persona synthesis
               try:
                   _log("info", "━━━ STAGE 3: Persona Synthesis ━━━")
                   fresh_quality = analyze_quality(load_all_reviews(selected_pid))
                   run_persona_synthesis(selected_pid, fresh_quality, log_fn=_log)
                   _log("ok", "Persona synthesis complete!")
                   proj = load_project(selected_pid)
               except Exception as e:
                   status_box.update(label="Persona synthesis failed", state="error")
                   st.error(f"Synthesis failed: {e}")
                   st.exception(e)
                   st.stop()

               status_box.update(label="Pipeline complete! ✓", state="complete")
               st.rerun()

    # ===========================================================================
    # SECTION 7: Outputs
    # ===========================================================================
    od = outputs_dir(selected_pid)
    stem = _output_stem(selected_pid)
    json_out = od / f"{stem}_persona.json"

    if json_out.exists():
       st.divider()
       st.markdown("## Outputs")

       persona = json.loads(json_out.read_text())
       built = datetime.fromtimestamp(json_out.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
       st.success(f"Persona built: {built} | {quality['total']:,} reviews | Quality: **{quality['score'].upper()}**")

       # Quick persona summary
       seg = persona.get("segment", {})
       ep = persona.get("emotional_profile", {})
       c1, c2, c3 = st.columns(3)
       c1.metric("Patience", f"{ep.get('baseline_patience',0.5):.0%}")
       c2.metric("Starting Trust", f"{ep.get('trust_starting_point',0.5):.0%}")
       c3.metric("Behavioral Rules", len(persona.get("behavioral_rules", [])))

       # Voice sample
       st.markdown("**Voice:**")
       st.info(persona.get("voice_sample", "")[:400] + ("..." if len(persona.get("voice_sample","")) > 400 else ""), icon="🗣️")

       # File downloads
       col1, col2, col3 = st.columns(3)
       with col1:
           fn_json = f"{stem}_persona.json"
           st.markdown(f"**{fn_json}**")
           st.caption("Structured config for AI agents")
           st.download_button("⬇️ Download", json_out.read_bytes(), fn_json, "application/json", use_container_width=True)
           with st.expander("Preview"):
               st.json(persona)

       with col2:
           fn_md = f"{stem}_persona.md"
           md_out = od / fn_md
           st.markdown(f"**{fn_md}**")
           st.caption("Rich narrative — use as system prompt context")
           if md_out.exists():
               st.download_button("⬇️ Download", md_out.read_bytes(), fn_md, "text/markdown", use_container_width=True)
               with st.expander("Preview"):
                   st.markdown(md_out.read_text())

       with col3:
           fn_rag = f"{stem}_rag_index.jsonl"
           rag_out = od / fn_rag
           st.markdown(f"**{fn_rag}**")
           st.caption("Retrieval index for RAG-augmented agents")
           if rag_out.exists():
               lines = rag_out.read_text().strip().split("\n")
               st.download_button("⬇️ Download", rag_out.read_bytes(), fn_rag, "application/x-ndjson", use_container_width=True)
               with st.expander(f"Preview ({len(lines)} entries)"):
                   for line in lines[:8]:
                       e = json.loads(line)
                       st.markdown(f"**[{e.get('trait_label','')}]** {e.get('text','')[:120]}")
                       st.divider()
