# Proxi AI — Session Log

---

## 2026-04-03 (Session: Pipeline schema compatibility fix)

Fixed downstream schema mismatch caused by new CI-annotated clustering output (`_count_with_confidence` produces `list[dict]` instead of `list[str]`). Added `_signal_text`/`_signal_texts` helpers for format-agnostic signal extraction, added cluster weight validation and missing-field backfill, exposed `random_state` seed control in UI, and hardened `pipeline/export.py` against malformed batch data. Both old and new cluster formats validated end-to-end.

---

## 2026-04-02 (Session: Training dashboard + bug fixes)

**Reviewed** the full codebase built by the previous Claude agent session — no changes made, just oriented on current state.

Built `train.py` — a focused Streamlit training dashboard for creating demographic persona context files from review data (CSV upload + HuggingFace Amazon Reviews pull → signal extraction → clustering → persona synthesis → 3 output files). Also fixed `engine/knowledge.py` to support `persona.json` alongside `config.json` in persona directories, and updated `.env.example` with both API key docs.

---

2026-04-05: Added App Store + Play Store UI scraper sections, fixed "Run EVERYTHING" to force-pull all 5 sources (Amazon/Reddit/HN/AppStore/PlayStore), expanded step bar to 9 steps including new sources, added "← Home" back button on project page, added "↺ Rerun" button on queue cards (DONE and FAILED) with automatic output archiving before rerun.

2026-04-05 (2): Added "↺ Rerun All" button (archives done outputs + resets all to pending), added `requeue_all()` to queue_manager. Overhauled UI: new CSS with qcard system, animated status badges, live queue stats bar (pending/running/done/failed counts), job list no longer hidden in expander — renders inline below worker controls.

2026-04-05 (3): Raised DEFAULT_TOTAL to 8,000 and MAX_REVIEWS to 15,000. Removed "skip if already collected" guards in worker.py so reruns top up all sources. App Store and Play Store now use full source budget (not hardcoded 200/app). Added text-hash deduplication in append_reviews() so reruns never write duplicate reviews. Rerun/Rerun All buttons now update job config to target_total=8,000.
