"""SQLite storage layer for Proxi AI runs, personas, and insights.

Provides a thin wrapper around SQLite for persisting journey runs,
persona definitions, and generated insights.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS personas (
    persona_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    data_json    TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    persona_id     TEXT NOT NULL,
    target_url     TEXT NOT NULL,
    outcome        TEXT,
    outcome_reason TEXT,
    journey_json   TEXT NOT NULL,
    summary        TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (persona_id) REFERENCES personas(persona_id)
);

CREATE TABLE IF NOT EXISTS insights (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL UNIQUE,
    insights_json TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db(db_path: str = "proxi_runs.db") -> sqlite3.Connection:
    """Initialize the database and return a connection.

    Creates the ``personas``, ``runs``, and ``insights`` tables if they
    do not already exist.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        An open connection with ``row_factory`` set to
        ``sqlite3.Row`` for dict-like access.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Persona operations
# ---------------------------------------------------------------------------

def save_persona(db: sqlite3.Connection, persona: dict[str, Any]) -> str:
    """Persist a persona definition.

    If the persona dict already contains a ``"persona_id"`` key it will
    be used; otherwise a new UUID is generated.

    Parameters
    ----------
    db:
        An open database connection from :func:`init_db`.
    persona:
        The persona dict. Must contain at least ``"name"``.

    Returns
    -------
    str
        The persona_id (existing or newly generated).
    """
    persona_id: str = persona.get("persona_id") or str(uuid.uuid4())
    name: str = persona.get("name", "Unnamed Persona")

    db.execute(
        "INSERT OR REPLACE INTO personas (persona_id, name, data_json) "
        "VALUES (?, ?, ?)",
        (persona_id, name, json.dumps(persona)),
    )
    db.commit()
    return persona_id


def get_persona(db: sqlite3.Connection, persona_id: str) -> dict[str, Any] | None:
    """Retrieve a persona by ID.

    Returns
    -------
    dict or None
        The persona dict with ``persona_id``, ``name``, ``data_json``
        (parsed), and ``created_at``, or ``None`` if not found.
    """
    row = db.execute(
        "SELECT * FROM personas WHERE persona_id = ?", (persona_id,)
    ).fetchone()

    if row is None:
        return None

    return {
        "persona_id": row["persona_id"],
        "name": row["name"],
        "created_at": row["created_at"],
        **json.loads(row["data_json"]),
    }


def get_all_personas(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Retrieve all stored personas.

    Returns
    -------
    list[dict]
        A list of persona dicts, each structured like the output of
        :func:`get_persona`.
    """
    rows = db.execute(
        "SELECT * FROM personas ORDER BY created_at DESC"
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        results.append({
            "persona_id": row["persona_id"],
            "name": row["name"],
            "created_at": row["created_at"],
            **json.loads(row["data_json"]),
        })
    return results


# ---------------------------------------------------------------------------
# Run operations
# ---------------------------------------------------------------------------

def save_run(
    db: sqlite3.Connection,
    run_id: str,
    persona_id: str,
    target_url: str,
    outcome: str | None,
    outcome_reason: str | None,
    journey_json: str,
    summary: str | None,
) -> None:
    """Persist a journey run.

    Parameters
    ----------
    db:
        An open database connection from :func:`init_db`.
    run_id:
        Unique identifier for this run.
    persona_id:
        ID of the persona that performed the run.
    target_url:
        The URL the persona navigated.
    outcome:
        High-level outcome (e.g. ``"success"``, ``"failure"``).
    outcome_reason:
        Short explanation of the outcome.
    journey_json:
        The full journey log serialized as a JSON string.
    summary:
        Executive summary text, or ``None``.
    """
    db.execute(
        "INSERT OR REPLACE INTO runs "
        "(run_id, persona_id, target_url, outcome, outcome_reason, "
        "journey_json, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, persona_id, target_url, outcome, outcome_reason,
         journey_json, summary),
    )
    db.commit()


def get_run(db: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Retrieve a single run by ID.

    Returns
    -------
    dict or None
        The run dict with parsed ``journey_json``, or ``None``.
    """
    row = db.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()

    if row is None:
        return None

    return _row_to_run(row)


def get_runs_for_persona(
    db: sqlite3.Connection,
    persona_id: str,
) -> list[dict[str, Any]]:
    """Retrieve all runs for a given persona.

    Returns
    -------
    list[dict]
        Runs ordered by creation time (newest first).
    """
    rows = db.execute(
        "SELECT * FROM runs WHERE persona_id = ? ORDER BY created_at DESC",
        (persona_id,),
    ).fetchall()

    return [_row_to_run(row) for row in rows]


def get_all_runs(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Retrieve all stored runs.

    Returns
    -------
    list[dict]
        All runs ordered by creation time (newest first).
    """
    rows = db.execute(
        "SELECT * FROM runs ORDER BY created_at DESC"
    ).fetchall()

    return [_row_to_run(row) for row in rows]


# ---------------------------------------------------------------------------
# Insight operations
# ---------------------------------------------------------------------------

def save_insights(
    db: sqlite3.Connection,
    run_id: str,
    insights_json: str,
) -> None:
    """Persist generated insights for a run.

    Parameters
    ----------
    db:
        An open database connection from :func:`init_db`.
    run_id:
        The run ID these insights belong to.
    insights_json:
        The insights dict serialized as a JSON string.
    """
    db.execute(
        "INSERT OR REPLACE INTO insights (run_id, insights_json) "
        "VALUES (?, ?)",
        (run_id, insights_json),
    )
    db.commit()


def get_insights(
    db: sqlite3.Connection,
    run_id: str,
) -> dict[str, Any] | None:
    """Retrieve insights for a run.

    Returns
    -------
    dict or None
        The parsed insights dict, or ``None`` if no insights exist.
    """
    row = db.execute(
        "SELECT * FROM insights WHERE run_id = ?", (run_id,)
    ).fetchone()

    if row is None:
        return None

    return {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        **json.loads(row["insights_json"]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a database row into a run dict with parsed JSON."""
    return {
        "run_id": row["run_id"],
        "persona_id": row["persona_id"],
        "target_url": row["target_url"],
        "outcome": row["outcome"],
        "outcome_reason": row["outcome_reason"],
        "journey": json.loads(row["journey_json"]),
        "summary": row["summary"],
        "created_at": row["created_at"],
    }
