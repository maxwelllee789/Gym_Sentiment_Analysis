"""
SQLite database for Gym Comparator AI.
Stores analysis results, comparison history, and provides query helpers.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "gym.db")


# ============================================================
# CONNECTION
# ============================================================
def get_connection():
    """Get SQLite connection with row factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================
# SCHEMA INIT
# ============================================================
def init_db():
    """Create tables and indexes if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ---- Table: gyms ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gyms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gmaps_link TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            address TEXT,
            rating REAL,
            review_count INTEGER,
            photo_url TEXT,
            lat REAL,
            lng REAL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---- Table: analyses ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gym_id INTEGER NOT NULL,
            positif_pct REAL NOT NULL,
            netral_pct REAL NOT NULL,
            negatif_pct REAL NOT NULL,
            total_sentences INTEGER,
            pros_json TEXT,
            cons_json TEXT,
            kesimpulan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (gym_id) REFERENCES gyms(id) ON DELETE CASCADE
        )
    """)

    # ---- Table: comparisons ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gym_a_id INTEGER NOT NULL,
            gym_b_id INTEGER NOT NULL,
            analysis_a_id INTEGER NOT NULL,
            analysis_b_id INTEGER NOT NULL,
            winner TEXT NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (gym_a_id) REFERENCES gyms(id) ON DELETE CASCADE,
            FOREIGN KEY (gym_b_id) REFERENCES gyms(id) ON DELETE CASCADE,
            FOREIGN KEY (analysis_a_id) REFERENCES analyses(id) ON DELETE CASCADE,
            FOREIGN KEY (analysis_b_id) REFERENCES analyses(id) ON DELETE CASCADE
        )
    """)

    # ---- Indexes ----
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_gym ON analyses(gym_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_date ON analyses(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_date ON comparisons(created_at)")

    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {DB_PATH}")


# ============================================================
# GYMS
# ============================================================
def upsert_gym(info: dict) -> int:
    """
    Insert new gym or update existing one (by gmaps_link).
    Returns gym ID.
    """
    link = info.get("gmaps_link", "").strip()
    if not link:
        raise ValueError("gmaps_link is required")

    conn = get_connection()
    cursor = conn.cursor()

    # Check existing
    cursor.execute("SELECT id FROM gyms WHERE gmaps_link = ?", (link,))
    row = cursor.fetchone()

    if row:
        gym_id = row["id"]
        cursor.execute("""
            UPDATE gyms SET
                name = ?,
                address = ?,
                rating = ?,
                review_count = ?,
                photo_url = ?,
                lat = ?,
                lng = ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            info.get("name", "-"),
            info.get("address", "-"),
            info.get("rating", 0),
            info.get("review_count", 0),
            info.get("photo_url"),
            info.get("lat"),
            info.get("lng"),
            gym_id,
        ))
    else:
        cursor.execute("""
            INSERT INTO gyms (gmaps_link, name, address, rating, review_count,
                              photo_url, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            link,
            info.get("name", "-"),
            info.get("address", "-"),
            info.get("rating", 0),
            info.get("review_count", 0),
            info.get("photo_url"),
            info.get("lat"),
            info.get("lng"),
        ))
        gym_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return gym_id


def get_gym_by_link(gmaps_link: str):
    """Fetch gym record by its Google Maps link."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM gyms WHERE gmaps_link = ?", (gmaps_link,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_gyms():
    """Fetch all gyms, ordered by name."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM gyms ORDER BY name ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# ANALYSES
# ============================================================
def save_analysis(gym_id: int, analysis: dict) -> int:
    """Save a sentiment analysis result. Returns analysis ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses
        (gym_id, positif_pct, netral_pct, negatif_pct,
         total_sentences, pros_json, cons_json, kesimpulan)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gym_id,
        float(analysis.get("positif_pct", 0)),
        float(analysis.get("netral_pct", 0)),
        float(analysis.get("negatif_pct", 0)),
        int(analysis.get("total_sentences", 0)),
        json.dumps(analysis.get("pros_by_category", {}), ensure_ascii=False),
        json.dumps(analysis.get("cons_by_category", {}), ensure_ascii=False),
        analysis.get("kesimpulan", ""),
    ))

    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id


def get_analysis(analysis_id: int):
    """Fetch single analysis by ID, decode JSON fields."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    record = dict(row)
    record["pros_by_category"] = json.loads(record.pop("pros_json") or "{}")
    record["cons_by_category"] = json.loads(record.pop("cons_json") or "{}")
    return record


def get_latest_analysis(gym_id: int):
    """Fetch most recent analysis for a gym."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE gym_id = ? ORDER BY created_at DESC LIMIT 1",
        (gym_id,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    record = dict(row)
    record["pros_by_category"] = json.loads(record.pop("pros_json") or "{}")
    record["cons_by_category"] = json.loads(record.pop("cons_json") or "{}")
    return record


# ============================================================
# COMPARISONS
# ============================================================
def save_comparison(
    gym_a_id: int,
    gym_b_id: int,
    analysis_a_id: int,
    analysis_b_id: int,
    rekom: dict,
) -> int:
    """Save comparison record. Returns comparison ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO comparisons
        (gym_a_id, gym_b_id, analysis_a_id, analysis_b_id, winner, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        gym_a_id,
        gym_b_id,
        analysis_a_id,
        analysis_b_id,
        rekom.get("pemenang", "-"),
        rekom.get("alasan", ""),
    ))

    conn.commit()
    comparison_id = cursor.lastrowid
    conn.close()
    return comparison_id


def get_recent_comparisons(limit: int = 20):
    """Fetch recent comparisons with gym names joined."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            c.id,
            c.winner,
            c.reason,
            c.created_at,
            ga.name AS gym_a_name,
            gb.name AS gym_b_name,
            a1.positif_pct AS a_positif,
            a2.positif_pct AS b_positif
        FROM comparisons c
        JOIN gyms ga ON ga.id = c.gym_a_id
        JOIN gyms gb ON gb.id = c.gym_b_id
        JOIN analyses a1 ON a1.id = c.analysis_a_id
        JOIN analyses a2 ON a2.id = c.analysis_b_id
        ORDER BY c.created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# STATS & MAINTENANCE
# ============================================================
def get_stats() -> dict:
    """Return summary statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    total_gyms = cursor.execute("SELECT COUNT(*) FROM gyms").fetchone()[0]
    total_analyses = cursor.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_comparisons = cursor.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]

    avg_rating = cursor.execute(
        "SELECT AVG(rating) FROM gyms WHERE rating > 0"
    ).fetchone()[0]

    conn.close()
    return {
        "total_gyms": total_gyms,
        "total_analyses": total_analyses,
        "total_comparisons": total_comparisons,
        "avg_gym_rating": round(avg_rating, 2) if avg_rating else 0,
    }


def cleanup_old_analyses(days: int = 30) -> int:
    """Delete analyses older than N days. Returns count deleted."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analyses WHERE created_at < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def reset_db():
    """Drop all tables and recreate. DANGER: wipes all data."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS comparisons")
    conn.execute("DROP TABLE IF EXISTS analyses")
    conn.execute("DROP TABLE IF EXISTS gyms")
    conn.commit()
    conn.close()
    init_db()
    print("[DB] Reset complete")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import sys

    init_db()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "stats":
            stats = get_stats()
            print("\n=== DATABASE STATS ===")
            for k, v in stats.items():
                print(f"  {k:25s} : {v}")
        elif cmd == "history":
            print("\n=== RECENT COMPARISONS ===")
            for c in get_recent_comparisons(10):
                print(f"  [{c['created_at']}] {c['gym_a_name']} vs {c['gym_b_name']} → {c['winner']}")
        elif cmd == "cleanup":
            n = cleanup_old_analyses(30)
            print(f"Deleted {n} old analyses")
        elif cmd == "reset":
            confirm = input("Wipe all data? (yes/no): ")
            if confirm.lower() == "yes":
                reset_db()
            else:
                print("Cancelled")
        else:
            print(f"Unknown command: {cmd}")
            print("Available: stats, history, cleanup, reset")
    else:
        print("DB initialized. Available commands:")
        print("  python db.py stats")
        print("  python db.py history")
        print("  python db.py cleanup")
        print("  python db.py reset")