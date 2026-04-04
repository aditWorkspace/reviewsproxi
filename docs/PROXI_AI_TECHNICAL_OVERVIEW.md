# Proxi AI — Technical & Strategic Overview

## What It Is

Proxi AI is a synthetic user persona engine. It takes a demographic description (e.g. "college students aged 18-24"), builds a statistically grounded behavioral persona from real consumer data, then deploys that persona as an autonomous browser agent to test websites. The agent thinks, browses, reacts, and makes decisions like a real person from that demographic would — and produces a full journey trace that tells you exactly where your website fails for that audience.

The core thesis: instead of hiring 50 real users for UX testing, you train synthetic personas on thousands of real reviews and Reddit comments, then run them against your website in minutes.

---

## The Full Pipeline (6 Stages)

### Stage 0: Demographic Intelligence

**Input:** A natural language description like "US college students aged 18-24, extremely price-sensitive, shop on mobile late at night, influenced by TikTok and peer recommendations"

**What happens:** Claude Sonnet maps this description to:
- **18 specific Amazon products** where 90%+ of buyers ARE this demographic (e.g. "College Ruled Composition Notebooks 6-Pack", not "iPhone 15" which everyone buys)
- **15 subreddits** where the majority of active posters are this demographic
- **Demographic profile** with core keywords, exclusion keywords, age range, and motivations

**Why this matters:** This is the only step where LLM "creativity" is intentional. We need it to bridge from a human description to specific, queryable data sources. Everything downstream is data-driven.

**Output:** `intelligence.json` — product list with accuracy scores, subreddit list with relevance scores, keyword sets for filtering.

---

### Stage 1: Amazon Review Collection

**Source:** HuggingFace McAuley-Lab/Amazon-Reviews-2023 (571M reviews)

**How it works:** For each product identified in Stage 0, we stream reviews from the HuggingFace dataset and match them using keyword overlap:
- `review_keywords` — words likely in the review text for that product
- `context_keywords` — demographic-specific language ("dorm", "campus", "semester")
- `exclusion_keywords` — signals the reviewer is NOT our demographic ("my kids", "retirement")

A review matches if it has 1+ review keyword hits OR 2+ context/demographic keyword hits, with zero exclusion hits.

**Why Amazon is the primary source:** Amazon reviews are product-specific purchase feedback. The reviewer actually bought the thing. They include ratings, verified purchase status, and detailed text about their experience. This is the highest quality behavioral signal we can get.

Each matched review is tagged with the product's `accuracy` score from the intelligence phase. This feeds into extraction weighting later.

**Typical yield:** 1,000-3,000 reviews across 12-18 products.

---

### Stage 2: Reddit + CSV Data (Supplementary)

**Reddit scraping** adds broader behavioral context:
1. Fetch top threads from each subreddit via Reddit JSON API
2. LLM-filter threads for demographic relevance (scores each thread title)
3. Scrape comments from relevant threads, deduplicating recurring weekly threads
4. Per-comment keyword scoring using the demographic profile's core/exclusion keywords
5. Combined accuracy = `thread_relevance × comment_demographic_score`

**Why Reddit is discounted (0.7x):** Reddit comments are discussion-based, not purchase-focused reviews. A comment in r/college about laptop frustrations is useful context, but it's noisier than an Amazon review from someone who actually bought a laptop case. In the extraction step, Reddit's effective accuracy is multiplied by 0.7, meaning fewer Reddit comments qualify for the high-confidence 2x boost.

**CSV upload** supports G2, Trustpilot, or custom review data. Flexible column mapping handles most formats.

**Why multi-source matters:** Single-source data creates platform bias. Amazon reviews over-index on product satisfaction; Reddit over-indexes on frustration and social dynamics. Cross-platform signals are the most reliable — if "hidden pricing" shows up as a pain point in both Amazon reviews AND Reddit threads, it's a real pattern, not platform noise.

---

### Stage 3: Signal Extraction (LLM-Powered)

**What happens:** Reviews are batched (~30 per batch) and sent to the LLM for structured signal extraction. Each batch returns:
- `pain_points` — what frustrates this demographic
- `desired_outcomes` — what they want
- `purchase_triggers` — what makes them buy
- `deal_breakers` — what makes them leave immediately
- `objections` — what makes them hesitate
- `friction_tolerance` — low/medium/high

**Source-aware weighting:**
- Reviews with accuracy >= 0.85 are duplicated in the batch (2x weight) so the LLM sees them more
- Reddit reviews get a 0.7x discount on their accuracy before this threshold
- Reviews with accuracy < 0.2 are dropped entirely (off-demographic noise)

**Source-balanced batching:** Reviews are interleaved across sources (Amazon, Reddit, CSV) in round-robin order before batching, so each batch gets a mix. This prevents platform bias (e.g. 30 Amazon reviews in a row would bias the LLM's frequency estimates).

**Parallelism:** Uses ThreadPoolExecutor with up to 25 parallel workers across multiple API keys. Crash-safe — results are written to `signals.jsonl` incrementally.

**IMPORTANT:** The LLM extraction returns qualitative labels (pain point text, trigger descriptions). Any numeric scores the LLM returns (frequency, intensity) are NOT trusted downstream. Real frequencies come from counting how often a signal appears across batches in Stage 4.

**Typical yield:** 80-150 signal batches from 2,000-4,000 reviews.

---

### Stage 4: Clustering + Statistical Analysis

This is where raw data becomes statistics. Two key operations:

#### 4a: Embedding + KMeans Clustering

Signal batches are embedded using SentenceTransformer (`all-MiniLM-L6-v2`), then clustered with KMeans:
- Auto-optimized k: sweeps k=3..10, scores each with silhouette + intra-cluster similarity + inter-cluster distance
- Combined score = 0.5 × silhouette + 0.3 × intra_similarity + 0.2 × inter_distance
- Stability analysis: runs KMeans with 10 different random seeds, computes Adjusted Rand Index (ARI) across all pairs to verify cluster boundaries are stable, not random

Each cluster represents a behavioral trait group — a coherent set of signals that tend to co-occur.

#### 4b: Frequency Counting with Bootstrap Confidence Intervals

For each cluster, we count how often each signal appears and compute 95% CIs:

```
_count_with_confidence(items, n_batches, n_bootstrap=1000)
→ [{signal, count, frequency, ci_95_lower, ci_95_upper}, ...]
```

This is the statistical backbone. A pain point that appears in 40% of batches with CI [0.32, 0.48] is a strong signal. One that appears in 5% with CI [0.01, 0.09] is noise.

**Noise filtering:** Signals appearing in < 2 batches are excluded (likely one-off mentions, not real patterns).

**Source mix tracking:** Each cluster records what % of its data came from Amazon vs Reddit vs CSV. Single-source clusters get flagged — their signals may reflect platform culture, not demographic behavior.

**Per-cluster output:**
- Top 12 pain points, outcomes, triggers, deal breakers (with frequencies + CIs)
- Friction tolerance distribution (low/medium/high percentages)
- Source mix and single-source warnings
- Representative batches (closest to centroid)

---

### Stage 5: Data-Driven Pre-Computation

Before the persona synthesis LLM call, we compute key metrics directly from the cluster data:

#### Decision Weights (from signal frequencies)

Count how often each decision factor appears across all clusters, weighted by cluster size:
```
For each cluster:
  For each pain point and trigger:
    factor_counts[signal] += count × cluster_member_count
Normalize top 8 factors to sum to 1.0
```
These are REAL weights derived from what the data says matters most — not LLM-hallucinated numbers.

#### Emotional Profile (from friction/trust distributions)

- **baseline_patience:** Weighted average of friction_tolerance_distribution across clusters. `low` → 0.2, `medium` → 0.5, `high` → 0.8. A demographic with 60% "low" friction tolerance gets a patience of ~0.32.
- **trust_starting_point:** Ratio of purchase triggers to deal breakers. More breakers relative to triggers = lower starting trust. Range [0.2, 0.8].
- **frustration_decay:** Scaled from average pain points per cluster. More pain points = faster patience decay per friction event. Range [0.1, 0.4].

#### Vocabulary Set

All unique phrases from cluster representative batches — ensures the persona's voice uses REAL language from the data, not LLM-invented phrases.

---

### Stage 6: Persona Synthesis (LLM + Data Constraints)

The LLM receives:
- All cluster traits with frequencies
- Pre-computed decision weights, emotional profile, and vocabulary
- Data quality metrics
- Explicit instructions: "use these exact values, do NOT override them"

The LLM synthesizes the QUALITATIVE parts:
- Behavioral rules (translates data signals into testable agent actions)
- Voice sample (creates authentic inner monologue using data vocabulary)
- Trigger map (maps cluster signals to conversion/abandonment conditions)
- Purchase journey narrative

**Post-validation enforces data-driven values:**
1. `emotional_profile` is FORCED to match pre-computed values (LLM can't override)
2. `decision_weights` are validated: if sum != 1.0, re-normalized; if missing, replaced with pre-computed values
3. `deal_breakers` are checked against cluster data — reports what % are grounded vs invented
4. `_provenance` metadata stamps what's data-driven vs LLM-synthesized

**Output:** `persona.json` (structured config), `persona.md` (rich narrative), `rag_index.jsonl` (embedding index for RAG-augmented agent runs).

---

## The Agent System

### How the Agent Works

The persona JSON becomes a live agent that browses websites autonomously:

1. **Launch:** Playwright opens a headless Chromium browser, navigates to the target URL
2. **Observe:** Extracts page context — title, headings, body preview, interactive elements, pricing/signup/testimonial signals
3. **Decide:** Claude (as the persona) reads the page and decides what to do — click, scroll, type, navigate, wait, leave, or convert
4. **Act:** Playwright executes the action
5. **Update state:** Friction events decay patience (multiplicatively), positive events rebuild trust (additively with diminishing returns)
6. **Repeat:** Up to 25 steps or until the persona leaves/converts

### PersonaState — The Psychological Model

Two core meters drive behavior:

**Patience** (0.0 to 1.0): Decays multiplicatively on friction. `severity=0.7` → patience × 0.72. This means repeated friction compounds fast — three medium frictions can drain patience from 0.7 to 0.25. Patience barely recovers from positive events (+0.03 per positive × strength). Once annoyed, people don't forget.

**Trust** (0.0 to 1.0): Grows slowly on positive signals with diminishing returns: `trust += strength × 0.15 × (1 - trust)`. Gets nicked by friction: `trust -= severity × 0.1`. This means trust is hard to build and easy to lose — realistic.

**Conversion decision:** `combined_score = goal_satisfaction × 0.6 + trust × 0.4`. Converts if combined >= 0.65 AND trust >= 0.5. Both conditions must be met — you can't brute-force a conversion with features if trust is low.

**Deal breakers** cause immediate exit regardless of patience/trust: paywall with no trial, mandatory credit card, broken page, infinite loop, aggressive popup spam, data harvesting warning.

### What the Agent Sees

The browser manager extracts structured page context:
- Title, URL
- All headings (H1-H6)
- Body text preview (first ~3000 chars)
- Interactive elements: buttons, links, inputs with their text/labels
- Boolean signals: has_pricing, has_signup, has_testimonials

The agent prompt includes the persona description, current goals, deal breakers, and current psychological state (patience %, trust %, goals met so far). The LLM responds as the persona — including inner monologue, emotional state, friction/positive events noticed, and the chosen action.

### Journey Log — The Output

Every step is recorded:
```
step → url → observation → inner_monologue → goal_relevance → 
emotional_state → friction_events → positive_events → action → reasoning
```

Final output includes:
- Outcome: converted / abandoned / error
- Outcome reason (specific)
- Full step-by-step journey
- Final psychological state (patience, trust, goals met)
- All friction and positive events logged
- Screenshots and video (optional)

---

## The Insights Layer

### Single-Run Analysis

After an agent run, Claude analyzes the journey log and produces:
- **Root cause analysis** — what friction points happened and WHY
- **Conversion blockers** — ranked by severity (critical/high/medium/low)
- **Quick wins** — low-effort fixes with high expected impact
- **Competitive vulnerabilities** — where the site is at risk
- **Persona-specific insights** — what matters uniquely to THIS demographic

### Cross-Persona Comparison

Run multiple personas against the same site, then compare:
- **Universal friction** — problems ALL personas hit (fix these first)
- **Segment-specific friction** — problems only certain demographics hit
- **Conversion path divergence** — where different personas take different paths
- **Missed opportunities** — what the site could do but doesn't
- **Priority matrix** — impact × severity ranking for all issues

---

## The Strategic Edge

### Why This Approach Works

1. **Grounded in real data, not vibes.** Decision weights come from counting signal frequencies across thousands of reviews with bootstrap confidence intervals. The persona's patience level comes from friction tolerance distributions, not a prompt that says "make them impatient."

2. **Multi-source cross-validation.** Amazon reviews + Reddit comments + CSV data. Signals that appear across platforms are real. Single-source signals get flagged.

3. **Statistically rigorous.** Bootstrap 95% CIs on every signal frequency. Cluster stability measured via ARI. Source mix tracked per cluster. Quality scoring with objective thresholds.

4. **Reproducible.** Same demographic description → same intelligence → same reviews → same signals → same clusters → same persona. Cluster seeds are configurable for stability testing.

5. **Transparent provenance.** Every persona output is stamped with what's data-driven vs LLM-synthesized. You know exactly which numbers to trust.

6. **The agent is the test.** Instead of reading a persona doc and guessing how they'd react, you WATCH them react. The browser agent produces a step-by-step journey with inner monologue, emotional state changes, friction events, and a conversion decision — all traceable to the persona's data-driven profile.

### The Business Model

For any company that wants to optimize their website for a specific audience:
1. Describe the target demographic
2. Pipeline runs in ~30 minutes (data collection + extraction + clustering + synthesis)
3. Deploy the persona agent against their website
4. Get a detailed journey trace + insights report
5. Compare across multiple demographics to find universal vs segment-specific issues

This replaces expensive, slow user testing ($5K-50K per round, weeks of scheduling) with instant, repeatable, data-grounded synthetic testing.
