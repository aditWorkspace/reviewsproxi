"""Persona generation queue manager.

Provides pure functions for managing the persistent job queue stored at
projects/queue.json, plus helpers to spawn / stop the background worker.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUEUE_FILE = Path("projects/queue.json")
PID_FILE   = Path("projects/worker.pid")
LOGS_DIR   = Path("projects/queue_logs")

STATUS_PENDING  = "pending"
STATUS_RUNNING  = "running"
STATUS_DONE     = "done"
STATUS_FAILED   = "failed"
STATUS_SKIPPED  = "skipped"


# ---------------------------------------------------------------------------
# Queue CRUD
# ---------------------------------------------------------------------------

def read_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    try:
        return json.loads(QUEUE_FILE.read_text())
    except Exception:
        return []


def _write_queue(jobs: list[dict]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(jobs, indent=2, default=str))


def add_job(
    label: str,
    description: str,
    project_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict:
    """Append a new pending job and return it.

    Parameters
    ----------
    label:        Short demographic title (first line of the demographic description).
    description:  Full demographic description text.
    project_name: Company / project display name (e.g. "Disgo"). Used as the
                  project name in the UI and as the first segment of output filenames.
                  Defaults to label if not provided.
    config:       Optional overrides for target_total, max_threads, etc.
    """
    job_id = uuid.uuid4().hex[:8]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    job: dict[str, Any] = {
        "id": job_id,
        "status": STATUS_PENDING,
        "project_name": project_name or label,
        "label": label,
        "description": description,
        "config": config or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "duration_s": None,
        "error": None,
        "persona_id": None,
        "log_file": str(LOGS_DIR / f"{job_id}.log"),
    }
    jobs = read_queue()
    jobs.append(job)
    _write_queue(jobs)
    return job


def update_job(job_id: str, **fields) -> None:
    """Patch fields on an existing job in-place."""
    jobs = read_queue()
    for j in jobs:
        if j["id"] == job_id:
            j.update(fields)
    _write_queue(jobs)


def remove_job(job_id: str) -> None:
    """Remove a job by id (only safe for pending jobs)."""
    _write_queue([j for j in read_queue() if j["id"] != job_id])


def clear_finished() -> None:
    """Remove all done/failed/skipped jobs from the queue."""
    _write_queue([
        j for j in read_queue()
        if j["status"] not in (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED)
    ])


def requeue_failed() -> int:
    """Reset failed jobs back to pending. Returns count reset."""
    jobs = read_queue()
    n = 0
    for j in jobs:
        if j["status"] == STATUS_FAILED:
            j["status"] = STATUS_PENDING
            j["error"] = None
            j["started_at"] = None
            j["finished_at"] = None
            n += 1
    _write_queue(jobs)
    return n


def requeue_all(target_total: int | None = None) -> int:
    """Reset all done + failed jobs back to pending. Returns count reset.

    If target_total is given, update that config value on every reset job —
    useful when rerunning with a higher review budget than the original run.
    """
    jobs = read_queue()
    n = 0
    for j in jobs:
        if j["status"] in (STATUS_DONE, STATUS_FAILED):
            j["status"] = STATUS_PENDING
            j["error"] = None
            j["started_at"] = None
            j["finished_at"] = None
            if target_total is not None:
                cfg = dict(j.get("config") or {})
                cfg["target_total"] = target_total
                j["config"] = cfg
            n += 1
    _write_queue(jobs)
    return n


# ---------------------------------------------------------------------------
# Worker process management
# ---------------------------------------------------------------------------

def is_worker_alive() -> bool:
    """True if the worker process is currently running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)   # signal 0 = existence check, no actual signal sent
        return True
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return False


def start_worker() -> int:
    """Spawn worker.py as a fully detached background process.

    Uses start_new_session=True so the worker survives browser / terminal
    disconnection.  Returns the PID.
    """
    worker_path = Path(__file__).resolve().parent.parent / "worker.py"
    proc = subprocess.Popen(
        [sys.executable, str(worker_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def stop_worker() -> bool:
    """Send SIGTERM to the worker. Returns True if a signal was sent."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        return True
    except Exception:
        PID_FILE.unlink(missing_ok=True)
        return False


def worker_pid() -> int | None:
    """Return current worker PID if alive, else None."""
    if not is_worker_alive():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def read_log(job_id: str, tail: int = 80) -> str:
    """Return the last *tail* lines of a job's log file."""
    log_file = LOGS_DIR / f"{job_id}.log"
    if not log_file.exists():
        return ""
    lines = log_file.read_text(errors="replace").splitlines()
    return "\n".join(lines[-tail:])
