#!/usr/bin/env python3
# Timestamp: "2026-02-09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_node_class.py
"""Semantic node classification for verification DAG nodes.

Five classes classify pipeline nodes by their role:
  - source:     Data acquisition scripts
  - input:      Raw data, configuration files
  - processing: Transform/analysis scripts
  - output:     Intermediate/final data products
  - claim:      Paper-level assertions (figures, statistics, text)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# Canonical node classes
NODE_CLASSES = ("source", "input", "processing", "output", "claim")

# File extensions → inferred node_class
_SCRIPT_EXTS = {".py", ".sh", ".r", ".R", ".jl", ".m"}
_DATA_EXTS = {
    ".csv",
    ".tsv",
    ".npy",
    ".npz",
    ".hdf5",
    ".h5",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".pkl",
    ".pickle",
    ".parquet",
    ".feather",
}
_FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tiff", ".eps"}
_TEX_EXTS = {".tex", ".bib", ".bbl"}


def infer_node_class(file_path: str, role: str) -> Optional[str]:
    """Infer node_class from file extension and session role.

    Parameters
    ----------
    file_path : str
        Path to the file.
    role : str
        Session role: 'input', 'output', or 'script'.

    Returns
    -------
    str or None
        Inferred node class, or None if ambiguous.
    """
    ext = Path(file_path).suffix.lower()

    if role == "script":
        return "source" if ext in _SCRIPT_EXTS else None

    if role == "input":
        if ext in _SCRIPT_EXTS:
            return "source"
        if ext in _DATA_EXTS:
            return "input"
        return "input"

    if role == "output":
        if ext in _DATA_EXTS:
            return "output"
        if ext in _FIGURE_EXTS:
            return "output"
        if ext in _TEX_EXTS:
            return "claim"
        return "output"

    return None


def migrate_add_node_class(db_path: Path) -> None:
    """Add node_class column to file_hashes table if not present.

    Safe to call multiple times (idempotent).

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("PRAGMA table_info(file_hashes)")
        columns = {row[1] for row in cursor.fetchall()}
        if "node_class" not in columns:
            conn.execute("ALTER TABLE file_hashes ADD COLUMN node_class TEXT")
            conn.commit()
    finally:
        conn.close()


def set_node_class(
    db_path: Path,
    session_id: str,
    file_path: str,
    node_class: str,
) -> None:
    """Set node_class for a specific file hash record.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database.
    session_id : str
        Session identifier.
    file_path : str
        Path to the file.
    node_class : str
        One of: source, input, processing, output, claim.
    """
    if node_class not in NODE_CLASSES:
        raise ValueError(
            f"Invalid node_class '{node_class}'. Must be one of: {NODE_CLASSES}"
        )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE file_hashes SET node_class = ? "
            "WHERE session_id = ? AND file_path = ?",
            (node_class, session_id, file_path),
        )
        conn.commit()
    finally:
        conn.close()


def auto_classify(db_path: Path) -> int:
    """Auto-classify all file_hashes records missing node_class.

    Returns
    -------
    int
        Number of records updated.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, file_path, role FROM file_hashes WHERE node_class IS NULL"
        ).fetchall()
        updated = 0
        for row_id, file_path, role in rows:
            nc = infer_node_class(file_path, role)
            if nc:
                conn.execute(
                    "UPDATE file_hashes SET node_class = ? WHERE id = ?",
                    (nc, row_id),
                )
                updated += 1
        conn.commit()
        return updated
    finally:
        conn.close()


__all__ = [
    "NODE_CLASSES",
    "infer_node_class",
    "migrate_add_node_class",
    "set_node_class",
    "auto_classify",
]
