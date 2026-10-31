from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("crimesense_reports.sqlite3")

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'Pending'
    )""")
    return conn

def submit_report(category: str, description: str, latitude: float, longitude: float):
    with _connect() as conn:
        conn.execute("INSERT INTO reports(created_at, category, description, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
                     (datetime.now(timezone.utc).isoformat(), category, description[:2000], latitude, longitude))

def get_reports():
    with _connect() as conn:
        return conn.execute("SELECT id, created_at, category, description, latitude, longitude, status FROM reports ORDER BY id DESC").fetchall()

def update_status(report_id: int, status: str):
    if status not in {"Pending", "Verified", "Rejected"}:
        raise ValueError("Invalid report status")
    with _connect() as conn:
        conn.execute("UPDATE reports SET status=? WHERE id=?", (status, int(report_id)))
