"""
Proxi AI — Agent Builder
Standalone client-facing app. Run with:
    streamlit run agent_builder.py --server.port 8502
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st

st.set_page_config(
    page_title="Agent Builder — Proxi",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide all Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stToolbar"]          { display: none !important; }

/* Page background */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"]              { background: #08090e !important; }

[data-testid="stMainBlockContainer"] {
    max-width: 640px;
    padding-top: 5vh;
    padding-bottom: 6rem;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #0e1018 !important;
    border: 1px solid #1e2035 !important;
    border-radius: 10px !important;
    color: #dde3f0 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 0.85rem !important;
    transition: border-color 0.18s !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #7c6cf7 !important;
    box-shadow: 0 0 0 3px rgba(124,108,247,0.1) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background: #7c6cf7 !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 0 !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 24px rgba(124,108,247,0.25) !important;
    transition: opacity 0.15s !important;
}
[data-testid="stFormSubmitButton"] > button:hover { opacity: 0.85 !important; }

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: #0e1018 !important;
    border: 1px solid #1e2035 !important;
    border-radius: 10px !important;
    color: #9d8ffc !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.65rem 0 !important;
    transition: border-color 0.15s, color 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #7c6cf7 !important;
    color: #c4b5fd !important;
}

/* ── Reset button ── */
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #1e2035 !important;
    border-radius: 10px !important;
    color: #4b5563 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: border-color 0.15s, color 0.15s !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #374151 !important;
    color: #9ca3af !important;
}

/* ── Alerts ── */
[data-testid="stAlert"]            { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
HEARTBEAT = Path("projects/worker_heartbeat.txt")
QUEUE_FILE = Path("projects/queue.json")
PROJECTS   = Path("projects")

STEP_ORDER = [
    ("intelligence", "Intelligence"),
    ("amazon",       "Amazon"),
    ("reddit",       "Reddit"),
    ("hackernews",   "Hacker News"),
    ("review_gate",  "Review Gate"),
    ("extraction",   "Extraction"),
    ("clustering",   "Clustering"),
    ("synthesis",    "Synthesis"),
    ("done",         "Complete"),
]

STEP_PCT = {
    "pending":        0,
    "create_project": 3,
    "intelligence":  10,
    "amazon":        24,
    "reddit":        40,
    "hackernews":    50,
    "review_gate":   64,
    "extraction":    78,
    "clustering":    88,
    "synthesis":     95,
    "done":         100,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def read_queue():
    if not QUEUE_FILE.exists():
        return []
    try:
        return json.loads(QUEUE_FILE.read_text())
    except Exception:
        return []

def find_job(job_id):
    for j in read_queue():
        if j["id"] == job_id:
            return j
    return None

def read_heartbeat():
    if not HEARTBEAT.exists():
        return {}
    try:
        return dict(
            line.split("=", 1)
            for line in HEARTBEAT.read_text().splitlines()
            if "=" in line
        )
    except Exception:
        return {}

def queue_position(job_id):
    count = 0
    for j in read_queue():
        if j["id"] == job_id:
            break
        if j["status"] == "pending":
            count += 1
    return count

def output_files(pid):
    d = PROJECTS / pid / "outputs"
    if not d.exists():
        return []
    files = []
    for pattern in ("*_persona.json", "*_persona.md", "*_rag_index.jsonl"):
        files += sorted(d.glob(pattern))
    return files

def word_count(s):
    return len(s.split())

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [("job_id", None), ("submitted", False), ("company", ""), ("label", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:3rem;">
  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;
              text-transform:uppercase;color:#7c6cf7;margin-bottom:0.5rem;">
    Proxi AI
  </div>
  <div style="font-size:1.9rem;font-weight:800;color:#f0f2f8;
              letter-spacing:-0.03em;line-height:1.15;margin-bottom:0.5rem;">
    Agent Builder
  </div>
  <div style="font-size:0.84rem;color:#374151;line-height:1.6;">
    Describe a user segment. We'll generate a complete persona context file<br>
    from real reviews, forums, and social data.
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Form ─────────────────────────────────────────────────────────────────────
if not st.session_state.submitted:
    with st.form("builder", clear_on_submit=False):
        company = st.text_input(
            "Company",
            placeholder="Notion, SailGTX, Disgo …",
            value=st.session_state.company,
        )
        label = st.text_input(
            "Demographic Label",
            placeholder="Enterprise Trade Compliance User (B2B SaaS)",
            value=st.session_state.label,
        )
        description = st.text_area(
            "Description",
            placeholder=(
                "Describe who this user is, what they do day-to-day, "
                "how they interact with the product, and what drives their feedback. "
                "Minimum 10 words."
            ),
            height=160,
        )

        wc = word_count(description)
        if description.strip():
            color = "#4ade80" if wc >= 10 else "#f87171"
            st.markdown(
                f'<div style="font-size:0.72rem;color:{color};'
                f'margin-top:-0.5rem;margin-bottom:0.75rem;">'
                f'{wc} word{"s" if wc != 1 else ""}'
                f'{" ✓" if wc >= 10 else f" — need {10 - wc} more"}'
                f'</div>',
                unsafe_allow_html=True,
            )

        go = st.form_submit_button("Generate Persona →")

    if go:
        errors = []
        if not company.strip():   errors.append("Company name is required.")
        if not label.strip():     errors.append("Demographic label is required.")
        if word_count(description) < 10: errors.append("Description must be at least 10 words.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            from data.queue_manager import add_job, is_worker_alive, start_worker
            job = add_job(label.strip(), description.strip(), project_name=company.strip())
            if not is_worker_alive():
                start_worker()
            st.session_state.job_id    = job["id"]
            st.session_state.submitted = True
            st.session_state.company   = company.strip()
            st.session_state.label     = label.strip()
            st.rerun()

# ─── Progress / result ────────────────────────────────────────────────────────
else:
    job_id = st.session_state.job_id
    job    = find_job(job_id)

    # ── Divider + context ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="font-size:0.78rem;color:#374151;margin-bottom:1.75rem;">
        {st.session_state.company}
        <span style="color:#1e2035;margin:0 0.4rem;">·</span>
        {st.session_state.label}
    </div>
    """, unsafe_allow_html=True)

    if job is None:
        st.error("Job not found — it may have been removed from the queue.")

    elif job["status"] == "failed":
        err = job.get("error") or "Unknown error."
        st.markdown(f"""
        <div style="background:#100808;border:1px solid #3f1414;border-radius:12px;
                    padding:1.75rem 2rem;">
          <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.08em;
                      text-transform:uppercase;color:#f87171;margin-bottom:0.5rem;">
            Generation failed
          </div>
          <div style="font-size:0.83rem;color:#6b3030;line-height:1.6;">{err}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.button("Try again", type="primary"):
            from data.queue_manager import update_job, is_worker_alive, start_worker
            update_job(job_id, status="pending", error=None, started_at=None, finished_at=None)
            if not is_worker_alive():
                start_worker()
            st.rerun()

    elif job["status"] == "done":
        pid   = job.get("persona_id") or f"queue_{job_id}"
        files = output_files(pid)

        # ── Done card ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#07110e;border:1px solid #14422a;border-radius:12px;
                    padding:1.75rem 2rem;margin-bottom:2rem;">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                      text-transform:uppercase;color:#4ade80;margin-bottom:0.6rem;">
            Complete
          </div>
          <div style="background:#0e1f18;border-radius:8px;height:6px;overflow:hidden;">
            <div style="width:100%;height:100%;
                        background:linear-gradient(90deg,#22c55e,#4ade80);
                        border-radius:8px;">
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Download buttons ─────────────────────────────────────────────────
        if files:
            FILE_META = {
                "json":  ("Persona JSON",   "Structured persona data"),
                "md":    ("Persona Doc",    "Human-readable persona"),
                "jsonl": ("RAG Index",      "Vector store entries"),
            }
            for f in sorted(files, key=lambda x: ["json","md","jsonl"].index(x.suffix.lstrip(".")) if x.suffix.lstrip(".") in ["json","md","jsonl"] else 9):
                ext = f.suffix.lstrip(".")
                title, subtitle = FILE_META.get(ext, (ext.upper(), f.name))
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:0.35rem;">
                  <div>
                    <div style="font-size:0.84rem;font-weight:600;color:#c8cfe0;">
                        {title}
                    </div>
                    <div style="font-size:0.72rem;color:#374151;">{subtitle}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    label=f"Download  ↓  .{ext}",
                    data=f.read_bytes(),
                    file_name=f.name,
                    mime="application/octet-stream",
                    key=f"dl_{f.name}",
                )
                st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        else:
            st.warning("Output files not found.")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.button("Build another →", type="secondary"):
            st.session_state.job_id    = None
            st.session_state.submitted = False
            st.session_state.company   = ""
            st.session_state.label     = ""
            st.rerun()

    else:
        # ── In progress (pending or running) ─────────────────────────────────
        hb      = read_heartbeat()
        hb_job  = hb.get("job", "")
        hb_step = hb.get("step", "pending")

        if job["status"] == "pending":
            pos       = queue_position(job_id)
            step_key  = "pending"
            pct       = 0
            status    = f"{pos} job{'s' if pos != 1 else ''} ahead in queue" if pos > 0 else "Starting…"
        else:
            step_key  = hb_step if hb_job == job_id else "create_project"
            pct       = STEP_PCT.get(step_key, 0)
            step_dict = dict(STEP_ORDER)
            status    = step_dict.get(step_key, step_key.replace("_", " ").title())

        # Progress bar
        st.markdown(f"""
        <div style="background:#0e1018;border:1px solid #1e2035;border-radius:12px;
                    padding:1.75rem 2rem;margin-bottom:1.5rem;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;
                      margin-bottom:1.1rem;">
            <div style="font-size:0.82rem;color:#6b7280;">{status}</div>
            <div style="font-size:0.78rem;font-weight:600;color:#7c6cf7;">{pct}%</div>
          </div>
          <div style="background:#13151f;border-radius:999px;height:5px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;border-radius:999px;
                        background:linear-gradient(90deg,#7c6cf7,#a78bfa);
                        transition:width 0.6s ease;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Step pills
        pills_html = '<div style="display:flex;gap:0.35rem;flex-wrap:wrap;">'
        reached = False
        for key, label_text in STEP_ORDER:
            if key == step_key:
                reached = True
                style = ("background:#1a1650;color:#a78bfa;"
                         "border:1px solid #4c1d95;")
            elif not reached:
                style = ("background:#071a10;color:#22c55e;"
                         "border:1px solid #14532d;")
            else:
                style = ("background:#0e1018;color:#1f2937;"
                         "border:1px solid #111827;")
            pills_html += (
                f'<span style="font-size:0.65rem;font-weight:600;padding:0.2rem 0.6rem;'
                f'border-radius:999px;letter-spacing:0.04em;text-transform:uppercase;'
                f'white-space:nowrap;{style}">{label_text}</span>'
            )
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

        time.sleep(3)
        st.rerun()
