#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_db.py
"""SQLite database for verification tracking."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scitex.config import get_paths


class VerificationDB:
    """
    SQLite database for tracking session runs and file hashes.

    Stores:
    - runs: session_id, script_path, timestamps, status
    - file_hashes: session_id, file_path, hash, role (input/script/output)
    - chains: parent-child relationships between sessions

    Examples
    --------
    >>> db = VerificationDB()
    >>> db.add_run("2025Y-11M-18D-09h12m03s_HmH5", "/path/script.py")
    >>> db.add_file_hash("2025Y-11M-18D-09h12m03s_HmH5", "data.csv", "a1b2c3", "input")
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize database connection.

        Parameters
        ----------
        db_path : str or Path, optional
            Path to database file. Defaults to ~/.scitex/verification.db
        """
        if db_path is None:
            db_path = get_paths().base / "verification.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    session_id TEXT PRIMARY KEY,
                    script_path TEXT,
                    script_hash TEXT,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    status TEXT,
                    exit_code INTEGER,
                    parent_session TEXT,
                    combined_hash TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS file_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES runs(session_id),
                    UNIQUE(session_id, file_path, role)
                );

                CREATE INDEX IF NOT EXISTS idx_file_path
                    ON file_hashes(file_path);
                CREATE INDEX IF NOT EXISTS idx_session
                    ON file_hashes(session_id);
                CREATE INDEX IF NOT EXISTS idx_role
                    ON file_hashes(role);
                CREATE INDEX IF NOT EXISTS idx_runs_status
                    ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_parent
                    ON runs(parent_session);

                CREATE TABLE IF NOT EXISTS verification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES runs(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_verification_session
                    ON verification_results(session_id);
            """
            )

    @contextmanager
    def _connect(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------

    def add_run(
        self,
        session_id: str,
        script_path: str,
        script_hash: Optional[str] = None,
        parent_session: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new run to the database.

        Parameters
        ----------
        session_id : str
            Unique session identifier
        script_path : str
            Path to the script that was run
        script_hash : str, optional
            Hash of the script file
        parent_session : str, optional
            Parent session ID for chain tracking
        metadata : dict, optional
            Additional metadata to store
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (session_id, script_path, script_hash, started_at, status,
                 parent_session, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    script_path,
                    script_hash,
                    datetime.now().isoformat(),
                    "running",
                    parent_session,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def finish_run(
        self,
        session_id: str,
        status: str = "success",
        exit_code: int = 0,
        combined_hash: Optional[str] = None,
    ) -> None:
        """
        Mark a run as finished.

        Parameters
        ----------
        session_id : str
            Session identifier
        status : str, optional
            Final status (success, failed, error)
        exit_code : int, optional
            Exit code of the script
        combined_hash : str, optional
            Combined hash of all inputs/outputs
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, status = ?, exit_code = ?, combined_hash = ?
                WHERE session_id = ?
                """,
                (
                    datetime.now().isoformat(),
                    status,
                    exit_code,
                    combined_hash,
                    session_id,
                ),
            )

    def set_parent(self, session_id: str, parent_session: str) -> None:
        """
        Set the parent session for a run.

        Parameters
        ----------
        session_id : str
            Session identifier
        parent_session : str
            Parent session identifier
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs SET parent_session = ? WHERE session_id = ?
                """,
                (parent_session, session_id),
            )

    def get_run(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get run information by session ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_runs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List runs with optional filtering.

        Parameters
        ----------
        status : str, optional
            Filter by status
        limit : int, optional
            Maximum number of results
        offset : int, optional
            Offset for pagination

        Returns
        -------
        list of dict
            List of run records
        """
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM runs
                    WHERE status = ?
                    ORDER BY started_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM runs
                    ORDER BY started_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # File hash operations
    # -------------------------------------------------------------------------

    def add_file_hash(
        self,
        session_id: str,
        file_path: str,
        hash_value: str,
        role: str,
    ) -> None:
        """
        Add a file hash record.

        Parameters
        ----------
        session_id : str
            Session identifier
        file_path : str
            Path to the file
        hash_value : str
            Hash of the file
        role : str
            Role of the file (input, script, output)
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_hashes
                (session_id, file_path, hash, role)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, file_path, hash_value, role),
            )

    def add_file_hashes(
        self,
        session_id: str,
        hashes: Dict[str, str],
        role: str,
    ) -> None:
        """
        Add multiple file hashes at once.

        Parameters
        ----------
        session_id : str
            Session identifier
        hashes : dict
            Mapping of file paths to hashes
        role : str
            Role of the files (input, script, output)
        """
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO file_hashes
                (session_id, file_path, hash, role)
                VALUES (?, ?, ?, ?)
                """,
                [(session_id, path, h, role) for path, h in hashes.items()],
            )

    def get_file_hashes(
        self,
        session_id: str,
        role: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Get file hashes for a session.

        Parameters
        ----------
        session_id : str
            Session identifier
        role : str, optional
            Filter by role

        Returns
        -------
        dict
            Mapping of file paths to hashes
        """
        with self._connect() as conn:
            if role:
                rows = conn.execute(
                    """
                    SELECT file_path, hash FROM file_hashes
                    WHERE session_id = ? AND role = ?
                    """,
                    (session_id, role),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT file_path, hash FROM file_hashes
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            return {row["file_path"]: row["hash"] for row in rows}

    def find_session_by_file(
        self,
        file_path: str,
        role: Optional[str] = None,
    ) -> List[str]:
        """
        Find sessions that used a specific file.

        Parameters
        ----------
        file_path : str
            Path to the file
        role : str, optional
            Filter by role (input, output)

        Returns
        -------
        list of str
            List of session IDs
        """
        with self._connect() as conn:
            if role:
                rows = conn.execute(
                    """
                    SELECT DISTINCT session_id FROM file_hashes
                    WHERE file_path = ? AND role = ?
                    ORDER BY recorded_at DESC
                    """,
                    (file_path, role),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT DISTINCT session_id FROM file_hashes
                    WHERE file_path = ?
                    ORDER BY recorded_at DESC
                    """,
                    (file_path,),
                ).fetchall()
            return [row["session_id"] for row in rows]

    # -------------------------------------------------------------------------
    # Chain operations
    # -------------------------------------------------------------------------

    def get_chain(self, session_id: str) -> List[str]:
        """
        Get the chain of parent sessions for a given session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        list of str
            List of session IDs from current to root
        """
        chain = [session_id]
        current = session_id

        with self._connect() as conn:
            while True:
                row = conn.execute(
                    "SELECT parent_session FROM runs WHERE session_id = ?",
                    (current,),
                ).fetchone()
                if not row or not row["parent_session"]:
                    break
                current = row["parent_session"]
                chain.append(current)

        return chain

    def get_children(self, session_id: str) -> List[str]:
        """Get child sessions that depend on this session."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id FROM runs
                WHERE parent_session = ?
                ORDER BY started_at
                """,
                (session_id,),
            ).fetchall()
            return [row["session_id"] for row in rows]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Verification result operations
    # -------------------------------------------------------------------------

    def record_verification(
        self,
        session_id: str,
        level: str,
        status: str,
    ) -> None:
        """
        Record a verification result.

        Parameters
        ----------
        session_id : str
            Session identifier
        level : str
            Verification level (cache, from_scratch)
        status : str
            Verification status (verified, mismatch, missing, unknown)
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO verification_results
                (session_id, level, status)
                VALUES (?, ?, ?)
                """,
                (session_id, level, status),
            )

    def get_latest_verification(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent verification result for a session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        dict or None
            Latest verification result with level, status, and timestamp
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT level, status, verified_at
                FROM verification_results
                WHERE session_id = ?
                ORDER BY verified_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_verification_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get verification history for a session.

        Parameters
        ----------
        session_id : str
            Session identifier
        limit : int, optional
            Maximum number of results

        Returns
        -------
        list of dict
            Verification results ordered by timestamp (newest first)
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT level, status, verified_at
                FROM verification_results
                WHERE session_id = ?
                ORDER BY verified_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._connect() as conn:
            total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            success_runs = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'success'"
            ).fetchone()[0]
            failed_runs = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'failed'"
            ).fetchone()[0]
            total_files = conn.execute("SELECT COUNT(*) FROM file_hashes").fetchone()[0]
            unique_files = conn.execute(
                "SELECT COUNT(DISTINCT file_path) FROM file_hashes"
            ).fetchone()[0]

            return {
                "total_runs": total_runs,
                "success_runs": success_runs,
                "failed_runs": failed_runs,
                "total_file_records": total_files,
                "unique_files": unique_files,
                "db_path": str(self.db_path),
            }


# Global instance
_DB_INSTANCE: Optional[VerificationDB] = None


def get_db() -> VerificationDB:
    """Get or create the global database instance."""
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = VerificationDB()
    return _DB_INSTANCE


# EOF
