"""
cache.py
--------
Persistent SQLite-backed cache for objective function evaluations.

Why a disk cache beats the in-memory dict
------------------------------------------
The in-memory `_cache` on ObjectiveFunction only lives for the duration of
one worker process.  With multiprocessing (n_workers > 1) each worker gets
its own Python interpreter and its own empty cache — so duplicate evaluations
across workers are not deduplicated, and nothing survives between runs.

The SQLite cache is a single file on disk that all worker processes open
concurrently (SQLite's WAL mode handles concurrent writers safely).  Results
from any worker are immediately visible to all others, and the cache persists
across runs so you never re-simulate an (x, propellant) pair you've already
evaluated.

Cache key
---------
SHA-256 of:
    propellant_key + "|" + comma-joined x values rounded to 4 decimal places

This matches the rounding in ObjectiveFunction.evaluate(), so a cached result
is returned whenever the optimizer revisits a point it already evaluated —
even across restarts.

Stored columns
--------------
    key         TEXT PRIMARY KEY   — hash
    tof_days    REAL
    fuel_kg     REAL
    cost_usd    REAL
    feasible    INTEGER            — 0 or 1
    status      TEXT
    x_repr      TEXT               — human-readable design vector (for inspection)
    propellant  TEXT
    evaluated_at TEXT              — ISO timestamp

Usage
-----
    from optimizer.cache import EvalCache
    cache = EvalCache("results/eval_cache.db")
    result = cache.get(x, propellant_key)
    if result is None:
        result = run_simulation(x)
        cache.put(x, propellant_key, result)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Thread-local storage so each thread keeps its own DB connection open
# (SQLite connections are not thread-safe across threads, but are fine within
# a single thread).
_local = threading.local()

# Default cache file location (relative to the project root)
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "eval_cache.db"
)

_ROUNDING = 4   # decimal places for cache key — must match ObjectiveFunction


def _make_key(x: np.ndarray, propellant_key: str) -> str:
    """SHA-256 hash of the design vector + propellant key."""
    rounded = ",".join(f"{v:.{_ROUNDING}f}" for v in x)
    raw = f"{propellant_key}|{rounded}".encode()
    return hashlib.sha256(raw).hexdigest()


class EvalCache:
    """
    Persistent evaluation cache backed by a single SQLite file.

    Thread-safe: each thread opens its own connection.
    Multi-process safe: WAL journal mode allows concurrent readers and
    one writer at a time without corruption.

    Parameters
    ----------
    db_path : path to the SQLite file.  Created (with parent directories)
              if it does not exist.
    """

    def __init__(self, db_path: str = DEFAULT_CACHE_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Create the table on first open (idempotent)
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                key          TEXT PRIMARY KEY,
                tof_days     REAL,
                fuel_kg      REAL,
                cost_usd     REAL,
                feasible     INTEGER,
                status       TEXT,
                x_repr       TEXT,
                propellant   TEXT,
                evaluated_at TEXT
            )
        """)
        conn.commit()

    # ── public interface ──────────────────────────────────────────────────────

    def get(self, x: np.ndarray, propellant_key: str) -> dict | None:
        """
        Return the cached result for (x, propellant_key), or None on a miss.

        The returned dict has the same keys as ObjectiveFunction.evaluate()
        except the raw phase results (p1/p2/p3_result), which are not stored.
        """
        key = _make_key(x, propellant_key)
        row = self._conn().execute(
            "SELECT tof_days, fuel_kg, cost_usd, feasible, status "
            "FROM evaluations WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        tof, fuel, cost, feasible, status = row
        return {
            "tof_days":  tof,
            "fuel_kg":   fuel,
            "cost_usd":  cost,
            "feasible":  bool(feasible),
            "status":    status,
            "p1_result": None,
            "p2_result": None,
            "p3_result": None,
        }

    def put(self, x: np.ndarray, propellant_key: str, result: dict) -> None:
        """
        Store a result.  Silently ignores duplicate keys (INSERT OR IGNORE).
        """
        key      = _make_key(x, propellant_key)
        x_repr   = ",".join(f"{v:.4f}" for v in x)
        now      = datetime.now(timezone.utc).isoformat()
        conn     = self._conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO evaluations
                (key, tof_days, fuel_kg, cost_usd, feasible, status, x_repr, propellant, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                float(result["tof_days"]),
                float(result["fuel_kg"]),
                float(result["cost_usd"]),
                int(result["feasible"]),
                str(result["status"]),
                x_repr,
                propellant_key,
                now,
            ),
        )
        conn.commit()

    def stats(self) -> dict:
        """Return a summary of what's in the cache."""
        conn = self._conn()
        total    = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        feasible = conn.execute("SELECT COUNT(*) FROM evaluations WHERE feasible=1").fetchone()[0]
        by_prop  = conn.execute(
            "SELECT propellant, COUNT(*) FROM evaluations GROUP BY propellant"
        ).fetchall()
        return {
            "total":      total,
            "feasible":   feasible,
            "by_propellant": dict(by_prop),
            "db_path":    self.db_path,
        }

    def clear(self) -> None:
        """Delete all cached results (useful for testing)."""
        conn = self._conn()
        conn.execute("DELETE FROM evaluations")
        conn.commit()

    # ── private ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return the thread-local connection, creating it if needed."""
        if not hasattr(_local, "connections"):
            _local.connections = {}
        if self.db_path not in _local.connections:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")   # concurrent-write safe
            conn.execute("PRAGMA synchronous=NORMAL")  # fast but safe
            _local.connections[self.db_path] = conn
        return _local.connections[self.db_path]


# ---------------------------------------------------------------------------
# Module-level singleton (shared within one process)
# ---------------------------------------------------------------------------

_default_cache: EvalCache | None = None


def get_default_cache(db_path: str = DEFAULT_CACHE_PATH) -> EvalCache:
    """Return the module-level singleton cache, creating it on first call."""
    global _default_cache
    if _default_cache is None or _default_cache.db_path != db_path:
        _default_cache = EvalCache(db_path)
    return _default_cache