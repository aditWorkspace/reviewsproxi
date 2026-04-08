# Proxi AI — Claude Instructions

## On session start
Read this file **once**. Do not re-read it during the session.
After making any meaningful change (new feature, model change, schema change, new file, bug fix), update the relevant section of this file before ending the session. Keep entries concise — this file is for orientation, not documentation.

---

## What this project is
Proxi AI generates synthetic user persona context documents from scraped review data.
Input: a demographic description (e.g. "Enterprise trade compliance user, B2B SaaS").
Output: three files per persona — `*_persona.json`, `*_persona.md`, `*_rag_index.jsonl`.

The main UI is a Streamlit app (`train.py`). There is no separate backend — all pipeline logic lives in `train.py` and `engine/`.

---

## Key files

| File | Purpose |
|---|---|
| `train.py` | Everything: Streamlit UI + all pipeline functions. ~3100 lines. |
| `worker.py` | Background queue worker. Imports train.py with `PROXI_WORKER_MODE=1` to skip UI. |
| `setup_overnight_queue.py` | One-shot script to bulk-add jobs to the queue. |
| `engine/llm.py` | OpenRouter client factory. Defines `MODEL` (DeepSeek, bulk extraction). |
| `engine/extract.py` | Signal extraction — batched LLM calls over review text. |
| `engine/aggregate.py` | Cluster deduplication, centroid-nearest canonical selection. |
| `engine/cluster.py` | KMeans with silhouette k-selection, negation-aware sentiment. |
| `data/queue_manager.py` | Queue CRUD + worker process management (start/stop/pid). |
| `data/budget.py` | Source budget controller (40% cap per source). |
| `data/sources/hackernews.py` | Algolia HN scraper (no API key needed). |
| `data/download.py` | Amazon review downloader with stratified sampling + reviewer dedup. |
| `projects/queue.json` | Persistent job queue (read fresh every poll cycle by worker). |
| `projects/worker.pid` | Worker PID file — used by UI to check if worker is alive. |
| `projects/worker_heartbeat.txt` | Written after each pipeline step — `job=<id>\nstep=<step>`. UI reads this for live progress. |

---

## Model routing

Defined in `train.py` (lines ~142–154):

```
MODEL            = "deepseek/deepseek-chat"      # bulk extraction (~100 batches/run)
MODEL_FILTER     = "qwen/qwen-plus"              # Reddit thread relevance filter
MODEL_SMART      = "anthropic/claude-sonnet-4"   # intelligence gen + persona synthesis
MODEL_INTELLIGENCE = MODEL_SMART                 # alias (keep existing call sites working)
```

**Rule:** DeepSeek for high-volume repeated calls. Sonnet for the two one-shot calls that determine output quality (intelligence generation + persona synthesis). Target cost: under $0.50 per demographic.

---

## Pipeline stages (in order)

1. **Intelligence** — LLM identifies relevant products + subreddits for the demographic (`MODEL_SMART`)
2. **Amazon reviews** — HuggingFace dataset pull with stratified sampling (low/mid/high star buckets: 15/8/77%)
3. **Reddit** — Pushshift/praw scrape with semantic comment scoring (sentence-transformers `all-MiniLM-L6-v2`)
4. **Hacker News** — Algolia API, no key required
5. **Review gate** — fail fast if < 500 reviews collected before expensive LLM steps
6. **Extraction** — batched signal extraction, 30 reviews/batch, parallel across API keys (`MODEL` / DeepSeek)
7. **Clustering** — KMeans with silhouette-score k-selection (sweep 3–10), negation-aware sentiment
8. **Synthesis** — single big LLM call producing structured `persona` JSON (`MODEL_SMART`)
9. **Rich markdown** — pure Python string templating from persona JSON, no LLM call

---

## Queue system

Jobs stored in `projects/queue.json`. Each job has:
- `project_name` — company name (e.g. "Disgo") → first segment of output filenames
- `label` — demographic title (first line of description) → second segment of output filenames
- `description` — full demographic paragraph
- `status` — pending / running / done / failed
- `persona_id` — set to `queue_{job_id}` when done (used to locate output files)

Output filename pattern: `{company_slug}_{label_slug}_persona.json` (built by `_output_stem(pid)` in train.py).

**Worker fail protections** (all in `worker.py`):
1. Stale-running reset on startup (crashed jobs → PENDING)
2. 2-hour per-job timeout via `signal.SIGALRM`
3. Amazon / Reddit / HN each wrapped in isolated try/except — one failing doesn't abort the job
4. Intel retry: 3 attempts with 15s/30s backoff
5. Review count gate: < 500 reviews → fail fast with clear message
6. Heartbeat file updated after every step for UI live progress

---

## Data sources

All free, no API keys required except OpenRouter:
- **Amazon**: HuggingFace `McAuley-Lab/Amazon-Reviews-2023` dataset
- **Reddit**: Pushshift via `data/sources/` (no key needed for read-only scraping)
- **Hacker News**: Algolia Search API (`https://hn.algolia.com/api/v1/search`)

Source budget: max 40% of total reviews from any single source. Target 3k reviews/job for overnight batch runs (keeps cost ~$0.40–$0.45/demographic).

---

## Running the app

```bash
# UI
streamlit run train.py

# Worker (overnight batch)
caffeinate -i python worker.py

# Load batch jobs
python setup_overnight_queue.py
```

---

## Important patterns

**PROXI_WORKER_MODE**: Set to `"1"` before importing train.py in worker context. Guards all Streamlit UI code behind `if not _WORKER_MODE:`. Also replaces `@st.cache_resource` with a no-op decorator.

**Resumability**: Every pipeline stage checks if its output already exists before running. Re-running a job that was interrupted picks up from the last completed stage.

**Project IDs**: Queue jobs use `pid = f"queue_{job_id}"` as a deterministic project ID. `new_project(company_name, label, desc, pid=pid)` accepts an explicit pid to avoid the default uuid-based generation.
