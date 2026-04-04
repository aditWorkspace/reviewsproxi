# Proxi AI — Session Log

---

## 2026-04-03 (Session: Pipeline schema compatibility fix)

Fixed downstream schema mismatch caused by new CI-annotated clustering output (`_count_with_confidence` produces `list[dict]` instead of `list[str]`). Added `_signal_text`/`_signal_texts` helpers for format-agnostic signal extraction, added cluster weight validation and missing-field backfill, exposed `random_state` seed control in UI, and hardened `pipeline/export.py` against malformed batch data. Both old and new cluster formats validated end-to-end.

---

## 2026-04-02 (Session: Training dashboard + bug fixes)

**Reviewed** the full codebase built by the previous Claude agent session — no changes made, just oriented on current state.

Built `train.py` — a focused Streamlit training dashboard for creating demographic persona context files from review data (CSV upload + HuggingFace Amazon Reviews pull → signal extraction → clustering → persona synthesis → 3 output files). Also fixed `engine/knowledge.py` to support `persona.json` alongside `config.json` in persona directories, and updated `.env.example` with both API key docs.

---
