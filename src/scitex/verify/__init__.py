#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/__init__.py
"""
SciTeX Verify Module - Hash-based verification for reproducible science.

This module provides tools to track, verify, and visualize the reproducibility
of scientific computations through cryptographic hashing.

Core Concepts
-------------
- **Run**: A single execution of a script with tracked inputs/outputs
- **Hash**: SHA256 fingerprint of files to detect changes
- **Chain**: Dependency links between runs (parent → child)
- **Verification**: Comparing stored hashes with current file states

Verification Levels
-------------------
- **verified-by-cache** (✓): Fast comparison of stored vs current hashes
- **verified-by-rerun** (✓✓): Full re-execution and comparison (slower, more thorough)

Examples
--------
>>> import scitex as stx

>>> # Automatic tracking via @stx.session + stx.io
>>> @stx.session
... def main():
...     data = stx.io.load("input.csv")   # Auto-tracked as input
...     result = process(data)
...     stx.io.save(result, "output.png") # Auto-tracked as output

>>> # Manual verification
>>> stx.verify.status()                   # Show changed files
>>> stx.verify.run("session_id")          # Verify specific run
>>> stx.verify.chain("output.png")        # Trace back to source

CLI Commands
------------
- ``scitex verify list`` - List all runs with verification status
- ``scitex verify run <id>`` - Verify specific run
- ``scitex verify chain <file>`` - Trace dependencies back to source
- ``scitex verify status`` - Show changed items (git status-like)

MCP Tools
---------
- ``verify_list`` - List runs
- ``verify_run`` - Verify specific run
- ``verify_chain`` - Trace chain
- ``verify_status`` - Show changes
"""

from __future__ import annotations

# Chain verification
from ._chain import (
    ChainVerification,
    FileVerification,
    RunVerification,
    VerificationLevel,
    VerificationStatus,
    get_status,
    verify_chain,
    verify_file,
    verify_run,
)

# Database
from ._db import (
    VerificationDB,
    get_db,
)

# Hash utilities
from ._hash import (
    combine_hashes,
    hash_directory,
    hash_file,
    hash_files,
    verify_hash,
)

# Integration hooks
from ._integration import (
    on_io_load,
    on_io_save,
    on_session_close,
    on_session_start,
)

# Rerun verification (separate module to avoid circular imports)
from ._rerun import verify_by_rerun, verify_run_from_scratch

# Tracker
from ._tracker import (
    SessionTracker,
    get_tracker,
    set_tracker,
    start_tracking,
    stop_tracking,
)

# Visualization
from ._visualize import (
    format_chain_verification,
    format_list,
    format_run_detailed,
    format_run_verification,
    format_status,
    generate_html_dag,
    generate_mermaid_dag,
    generate_plotly_dag,
    print_verification_summary,
    render_dag,
    render_plotly_dag,
)


# Convenience functions at module level
def list_runs(limit: int = 100, status: str = None):
    """List tracked runs."""
    db = get_db()
    return db.list_runs(status=status, limit=limit)


def status():
    """Get verification status summary (like git status)."""
    return get_status()


def run(session_id: str, from_scratch: bool = False):
    """Verify a specific run.

    Parameters
    ----------
    session_id : str
        Session identifier
    from_scratch : bool, optional
        If True, re-execute the script and verify outputs (slow but thorough).
        If False, only compare hashes (fast).
    """
    if from_scratch:
        return verify_run_from_scratch(session_id)
    return verify_run(session_id)


def chain(target: str):
    """Verify the chain for a target file."""
    return verify_chain(target)


def stats():
    """Get database statistics."""
    db = get_db()
    return db.stats()


__all__ = [
    # Hash utilities
    "hash_file",
    "hash_files",
    "hash_directory",
    "combine_hashes",
    "verify_hash",
    # Database
    "VerificationDB",
    "get_db",
    # Tracker
    "SessionTracker",
    "get_tracker",
    "set_tracker",
    "start_tracking",
    "stop_tracking",
    # Chain verification
    "VerificationStatus",
    "VerificationLevel",
    "FileVerification",
    "RunVerification",
    "ChainVerification",
    "verify_file",
    "verify_run",
    "verify_by_rerun",
    "verify_run_from_scratch",  # backward compat alias
    "verify_chain",
    "get_status",
    # Visualization
    "format_run_verification",
    "format_run_detailed",
    "format_chain_verification",
    "format_status",
    "format_list",
    "generate_mermaid_dag",
    "generate_html_dag",
    "generate_plotly_dag",
    "render_dag",
    "render_plotly_dag",
    "print_verification_summary",
    # Convenience functions
    "list_runs",
    "status",
    "run",
    "chain",
    "stats",
    # Integration hooks
    "on_session_start",
    "on_session_close",
    "on_io_load",
    "on_io_save",
]


# EOF
