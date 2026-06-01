from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "learners.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learner_state (
            user_id TEXT PRIMARY KEY,
            known_json TEXT NOT NULL,
            mistakes_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def get_state(user_id: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT known_json, mistakes_json FROM learner_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return {"user_id": user_id, "known": [], "mistakes": []}
    return {
        "user_id": user_id,
        "known": json.loads(row[0]),
        "mistakes": json.loads(row[1]),
    }


def save_state(user_id: str, known: list[str], mistakes: list[str]) -> dict:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO learner_state (user_id, known_json, mistakes_json)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                known_json = excluded.known_json,
                mistakes_json = excluded.mistakes_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, json.dumps(known), json.dumps(mistakes)),
        )
    return get_state(user_id)
