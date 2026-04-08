#!/usr/bin/env python3
"""
Proxi AI — Background Queue Worker
====================================
Processes persona generation jobs from projects/queue.json one at a time.
Designed to keep running even with the browser closed or the screen off.

Usage:
    python worker.py          # start worker (polls every 5s)
    caffeinate -i python worker.py   # recommended: prevents macOS deep sleep

NOTE: The worker survives with the screen off, but NOT if the Mac goes into
deep sleep (System Preferences → Battery → "Put hard disks to sleep").
Keep the laptop plugged in with display sleep allowed but computer sleep OFF,
or use caffeinate as shown above.

Fail-loop protections
---------------------
1. Stale-running reset  — on startup any job stuck in RUNNING is reset to
                          PENDING so it gets retried (handles prior crash).
2. Per-job timeout      — each job is killed (marked failed) after 2 hours
                          via SIGALRM so one hung job can't block the rest.
3. Isolated step errors — Amazon / Reddit / HN scraping failures are caught
                          individually; the job continues with whatever data
                          was collected rather than aborting entirely.
4. Intel retry          — intelligence generation is retried up to 3× with
                          backoff before the job is marked failed.
5. Review-count gate    — if total reviews collected is below MIN_REVIEWS
                          after all scraping, the job is marked failed with a
                          clear message before wasting LLM extraction quota.
6. Heartbeat file       — projects/worker_heartbeat.txt is updated after every
                          pipeline step so you can verify the worker is alive
                          (check mtime — if older than 30 min, something stalled).
"""

from __future__ import annotations

# ── Set BEFORE any other imports so train.py skips all Streamlit execution ──
import os
os.environ["PROXI_WORKER_MODE"] = "1"

import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Load .env so OPENROUTER_API_KEY etc. are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Queue management
from data.queue_manager import (
    LOGS_DIR,
    PID_FILE,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    read_queue,
    update_job,
)

# Pipeline functions — importable because PROXI_WORKER_MODE=1 skips UI code
import train as _t   # noqa: E402  (must come after env-var is set)

POLL_INTERVAL  = 5       # seconds between queue polls
JOB_TIMEOUT_S  = 7_200   # 2 hours — kill hung jobs before they block overnight run
MIN_REVIEWS    = 500     # below this, extraction/clustering are pointless
INTEL_RETRIES  = 3       # attempts for intelligence generation before failing job

HEARTBEAT_FILE = Path("projects/worker_heartbeat.txt")
QUEUE_FILE     = Path("projects/queue.json")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _make_logger(job_id: str):
    """Return a log_fn(kind, msg) that writes to the job's log file and stdout."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{job_id}.log"

    def log_fn(kind: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "→", "ok": "✓", "warn": "⚠", "data": "  "}.get(kind, " ")
        line = f"[{ts}] {prefix} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    return log_fn


# ---------------------------------------------------------------------------
# Fail protection helpers
# ---------------------------------------------------------------------------

def _heartbeat(job_id: str, step: str) -> None:
    """Update the heartbeat file — used to detect hangs externally."""
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_FILE.write_text(
        f"job={job_id}\nstep={step}\ntime={datetime.now().isoformat()}\n"
    )


def _reset_stale_running() -> int:
    """
    Protection 1 — Stale-running reset.

    If the worker previously crashed mid-job the job stays in RUNNING state
    forever, blocking the queue.  Reset every RUNNING job back to PENDING on
    startup so it gets retried from the beginning.
    """
    if not QUEUE_FILE.exists():
        return 0
    try:
        jobs = json.loads(QUEUE_FILE.read_text())
    except Exception:
        return 0

    n = 0
    for j in jobs:
        if j["status"] == STATUS_RUNNING:
            j["status"] = STATUS_PENDING
            j["started_at"] = None
            j["error"] = "Reset from stale RUNNING on worker restart"
            n += 1

    if n:
        QUEUE_FILE.write_text(json.dumps(jobs, indent=2, default=str))
        print(f"[worker] Reset {n} stale RUNNING job(s) → PENDING", flush=True)

    return n


class _JobTimeout(Exception):
    """Raised by SIGALRM when a job exceeds JOB_TIMEOUT_S."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _JobTimeout(f"Job exceeded {JOB_TIMEOUT_S // 60}-minute timeout")


# ---------------------------------------------------------------------------
# Single-job runner
# ---------------------------------------------------------------------------

def run_job(job: dict) -> None:
    job_id       = job["id"]
    label        = job["label"]           # demographic title (short)
    desc         = job["description"]     # full demographic description
    project_name = job.get("project_name", label)   # company name for display + filenames
    config       = job.get("config", {})

    target_total     = config.get("target_total", 5_000)
    max_threads      = config.get("max_threads", 8)
    max_comments     = config.get("max_comments", 150)
    hn_max_per_query = config.get("hn_max_per_query", 200)

    log = _make_logger(job_id)
    log("ok", f"━━━ Starting job [{job_id}]: {label} ━━━")

    # ── Protection 2: per-job timeout via SIGALRM ────────────────────────────
    # SIGALRM is Unix-only (macOS supported). If the platform lacks it we skip
    # silently rather than crashing.
    _alarm_supported = hasattr(signal, "SIGALRM")
    if _alarm_supported:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(JOB_TIMEOUT_S)

    started = datetime.now(timezone.utc).isoformat()
    update_job(job_id, status=STATUS_RUNNING, started_at=started)
    t0 = time.time()

    try:
        # ── 1. Create or retrieve project ────────────────────────────────────
        # pid is deterministic from job_id so we can always find/resume it.
        # project_name (company) becomes the display name; label (demographic
        # title) becomes the demographic_label — both end up in the output filenames.
        _heartbeat(job_id, "create_project")
        pid = f"queue_{job_id}"
        existing = _t.list_projects()
        if not any(p["id"] == pid for p in existing):
            log("info", f"Creating project '{pid}' ({project_name} — {label})")
            _t.new_project(project_name, label, desc, pid=pid)
        else:
            log("info", f"Resuming project '{pid}'")

        # ── 2. Generate intelligence — Protection 4: retry loop ──────────────
        _heartbeat(job_id, "intelligence")
        intel = _t.load_intelligence(pid)
        if not intel:
            for attempt in range(1, INTEL_RETRIES + 1):
                log("info", f"Generating intelligence (attempt {attempt}/{INTEL_RETRIES})…")
                try:
                    intel = _t.generate_intelligence(pid, desc, log_fn=log)
                    if intel:
                        break
                except Exception as exc:
                    log("warn", f"Intel attempt {attempt} failed: {exc}")
                    if attempt < INTEL_RETRIES:
                        time.sleep(15 * attempt)   # 15s, 30s backoff
            if not intel:
                raise RuntimeError(
                    f"Intelligence generation failed after {INTEL_RETRIES} attempts"
                )
        else:
            log("info", "Intelligence already exists — skipping")

        # ── 3. Amazon reviews — Protection 3: isolated step ──────────────────
        # Always runs — source_remaining() inside enforces the 40% budget cap,
        # so reruns only pull the delta needed to reach the new target_total.
        _heartbeat(job_id, "amazon")
        try:
            products = intel.get("products", [])
            selected = [p["name"] for p in products if p.get("accuracy", 0) >= 0.65][:6]
            if selected:
                log("info", f"Pulling Amazon reviews for {len(selected)} products…")
                _t.pull_amazon_reviews(
                    pid, intel, selected,
                    max_per_product=target_total // max(len(selected), 1),
                    target_total=target_total,
                    log_fn=log,
                )
            else:
                log("warn", "No high-accuracy products found — skipping Amazon")
        except Exception as exc:
            log("warn", f"Amazon scraping failed (continuing): {exc}")

        # ── 4. Reddit — Protection 3: isolated step ───────────────────────────
        # Always runs — source_remaining() caps naturally.
        _heartbeat(job_id, "reddit")
        try:
            subs = [s["name"] for s in sorted(
                intel.get("subreddits", []), key=lambda s: -s.get("relevance", 0)
            )[:8]]
            if subs:
                log("info", f"Scraping Reddit: {subs}…")
                _t.scrape_reddit_for_project(
                    pid, subs,
                    max_threads_per_sub=max_threads,
                    max_comments_per_thread=max_comments,
                    target_total=target_total,
                    log_fn=log,
                )
            else:
                log("warn", "No subreddits found — skipping Reddit")
        except Exception as exc:
            log("warn", f"Reddit scraping failed (continuing): {exc}")

        # ── 5. Hacker News — Protection 3: isolated step ─────────────────────
        # Always runs — source_remaining() caps naturally.
        _heartbeat(job_id, "hackernews")
        try:
            products   = intel.get("products", [])
            hn_queries = [p["name"] for p in products[:4] if p.get("accuracy", 0) >= 0.6]
            kws = intel.get("demographic_profile", {}).get("core_keywords", [])
            if kws:
                hn_queries.append(" ".join(kws[:3]))
            if hn_queries:
                log("info", f"Scraping HN: {hn_queries}…")
                _t.scrape_hn_for_project(
                    pid, hn_queries,
                    max_per_query=hn_max_per_query,
                    target_total=target_total,
                    log_fn=log,
                )
            else:
                log("warn", "No queries for HN — skipping")
        except Exception as exc:
            log("warn", f"HN scraping failed (continuing): {exc}")

        # ── 6. App Store — isolated step ─────────────────────────────────────
        # max_per_app is set to target_total so the only real cap is source_remaining()
        # (40% of target_total). This way budget is distributed across all matching apps
        # up to the full source ceiling, not arbitrarily capped at 200/app.
        _heartbeat(job_id, "appstore")
        try:
            products = intel.get("products", [])
            app_queries = [p["name"] for p in products[:6] if p.get("accuracy", 0) >= 0.55]
            if app_queries:
                log("info", f"Scraping App Store: {app_queries[:3]}…")
                _t.scrape_appstore_for_project(
                    pid, app_queries,
                    max_per_app=target_total,   # cap handled by source_remaining inside
                    target_total=target_total,
                    log_fn=log,
                )
            else:
                log("warn", "No products for App Store — skipping")
        except Exception as exc:
            log("warn", f"App Store scraping failed (continuing): {exc}")

        # ── 7. Play Store — isolated step ────────────────────────────────────
        _heartbeat(job_id, "playstore")
        try:
            products = intel.get("products", [])
            app_queries = [p["name"] for p in products[:6] if p.get("accuracy", 0) >= 0.55]
            if app_queries:
                log("info", f"Scraping Play Store: {app_queries[:3]}…")
                _t.scrape_playstore_for_project(
                    pid, app_queries,
                    max_per_app=target_total,   # cap handled by source_remaining inside
                    target_total=target_total,
                    log_fn=log,
                )
            else:
                log("warn", "No products for Play Store — skipping")
        except Exception as exc:
            log("warn", f"Play Store scraping failed (continuing): {exc}")

        # ── Protection 5: review-count gate ───────────────────────────────────
        _heartbeat(job_id, "review_gate")
        all_reviews = _t.load_all_reviews(pid)
        total_collected = len(all_reviews)
        log("info", f"Total reviews collected: {total_collected:,}")

        if total_collected < MIN_REVIEWS:
            raise RuntimeError(
                f"Only {total_collected} reviews collected (minimum {MIN_REVIEWS}). "
                "No suitable products/subreddits found for this demographic — "
                "refine the description and retry."
            )

        # ── 6. Signal extraction ──────────────────────────────────────────────
        _heartbeat(job_id, "extraction")
        log("info", "Running signal extraction…")
        n_batches = _t.run_extraction(pid, batch_size=30, force=False, log_fn=log)
        log("ok", f"Extraction: {n_batches} batches")

        # ── 7. Clustering ─────────────────────────────────────────────────────
        _heartbeat(job_id, "clustering")
        log("info", "Running clustering…")
        result = _t.run_clustering(
            pid,
            n_override=None,
            sweep_min=3,
            sweep_max=10,
            log_fn=log,
        )
        log("ok", f"Clustering: {result.get('chosen_n_clusters')} clusters")

        # ── 8. Persona synthesis ──────────────────────────────────────────────
        _heartbeat(job_id, "synthesis")
        log("info", "Synthesising persona…")
        quality = _t.analyze_quality(all_reviews)
        _t.run_persona_synthesis(pid, quality, log_fn=log)
        log("ok", "Persona synthesis complete")

    finally:
        # Always cancel the alarm whether we succeeded or not
        if _alarm_supported:
            signal.alarm(0)

    duration = round(time.time() - t0)
    _heartbeat(job_id, "done")
    log("ok", f"━━━ Job [{job_id}] done in {duration}s ━━━")
    update_job(
        job_id,
        status=STATUS_DONE,
        finished_at=datetime.now(timezone.utc).isoformat(),
        duration_s=duration,
        persona_id=pid,
    )


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Write PID so UI can check if we're alive
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f"[worker] Started. PID={os.getpid()}. Polling every {POLL_INTERVAL}s.", flush=True)
    print(f"[worker] Job timeout: {JOB_TIMEOUT_S // 60} min. Min reviews: {MIN_REVIEWS}.", flush=True)

    # Protection 1: reset any jobs that were stuck in RUNNING from a prior crash
    _reset_stale_running()

    try:
        while True:
            jobs = read_queue()
            pending = [j for j in jobs if j["status"] == STATUS_PENDING]

            if not pending:
                time.sleep(POLL_INTERVAL)
                continue

            job = pending[0]
            try:
                run_job(job)
            except KeyboardInterrupt:
                print("[worker] Interrupted — marking job as failed", flush=True)
                update_job(job["id"], status=STATUS_FAILED, error="Interrupted by user")
                break
            except _JobTimeout as exc:
                # Protection 2: timeout fired
                msg = str(exc)
                print(f"[worker] Job {job['id']} TIMED OUT: {msg}", flush=True)
                log_path = LOGS_DIR / f"{job['id']}.log"
                with open(log_path, "a") as fh:
                    fh.write(f"\n\nTIMEOUT: {msg}\n")
                update_job(
                    job["id"],
                    status=STATUS_FAILED,
                    error=f"Timeout: {msg}",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                time.sleep(5)   # brief pause before next job
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[worker] Job {job['id']} FAILED:\n{tb}", flush=True)
                log_path = LOGS_DIR / f"{job['id']}.log"
                with open(log_path, "a") as fh:
                    fh.write(f"\n\nFATAL ERROR:\n{tb}")
                update_job(
                    job["id"],
                    status=STATUS_FAILED,
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                # Brief pause before next job to avoid rapid failure loops
                time.sleep(10)

    finally:
        PID_FILE.unlink(missing_ok=True)
        HEARTBEAT_FILE.unlink(missing_ok=True)
        print("[worker] Exited cleanly.", flush=True)


if __name__ == "__main__":
    main()
