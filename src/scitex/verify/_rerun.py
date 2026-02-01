#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_rerun.py
"""Rerun verification - re-execute scripts and compare outputs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict

from ._chain import (
    FileVerification,
    RunVerification,
    VerificationLevel,
    VerificationStatus,
)
from ._db import get_db


def verify_by_rerun(
    target: str | list[str],
    timeout: int = 300,
    cleanup: bool = True,
) -> RunVerification | list[RunVerification]:
    """
    Verify session(s) by re-executing scripts and comparing outputs.

    Parameters
    ----------
    target : str or list[str]
        Session ID, script path, or artifact path.
        - run_id: directly use this run
        - script path: latest run that executed this script
        - artifact path: latest run which produced this file
    timeout : int, optional
        Maximum execution time in seconds (default: 300)
    cleanup : bool, optional
        Whether to remove the new session's output directory after verification

    Returns
    -------
    RunVerification or list[RunVerification]
        Single result if single target, list if multiple targets
    """
    if isinstance(target, list):
        return [_verify_single(t, timeout, cleanup) for t in target]
    return _verify_single(target, timeout, cleanup)


def _verify_single(
    target: str,
    timeout: int = 300,
    cleanup: bool = True,
) -> RunVerification:
    """Verify a single target."""
    db = get_db()

    # Resolve target to session_id
    session_id = _resolve_to_session_id(db, target)
    if not session_id:
        return _unknown_result(target, None)

    # Get original run info
    run_info = db.get_run(session_id)
    if not run_info:
        return _unknown_result(session_id, None)

    script_path = run_info.get("script_path")
    if not script_path or not Path(script_path).exists():
        return RunVerification(
            session_id=session_id,
            script_path=script_path,
            status=VerificationStatus.MISSING,
            files=[],
            combined_hash_expected=None,
            combined_hash_current=None,
            level=VerificationLevel.RERUN,
        )

    # Get expected output hashes from original session
    original_hashes = db.get_file_hashes(session_id, role="output")
    if not original_hashes:
        return _unknown_result(session_id, script_path)

    # Re-execute the script (creates new session)
    exec_result = _execute_script(script_path, timeout)
    if exec_result is not None:
        return exec_result._replace(session_id=session_id)

    # Find the new session (most recent from this script)
    new_session_id, new_sdir_run = _find_new_session(db, script_path, session_id)
    if not new_session_id:
        return _unknown_result(session_id, script_path)

    # Get new session's output hashes
    new_hashes = db.get_file_hashes(new_session_id, role="output")

    # Compare hashes by filename
    file_verifications = _compare_hashes(original_hashes, new_hashes)

    # Cleanup new session's output directory if requested
    if cleanup and new_sdir_run:
        _cleanup_session_dir(new_sdir_run)

    # Determine overall status
    status = _determine_status(file_verifications)

    # Record verification result in database for original session
    db.record_verification(
        session_id=session_id,
        level=VerificationLevel.RERUN.value,
        status=status.value,
    )

    return RunVerification(
        session_id=session_id,
        script_path=script_path,
        status=status,
        files=file_verifications,
        combined_hash_expected=run_info.get("combined_hash"),
        combined_hash_current=None,
        level=VerificationLevel.RERUN,
    )


def _resolve_to_session_id(db, target: str) -> str | None:
    """Resolve target to session_id.

    Accepts:
        - run_id: directly use this run
        - script path: latest run that executed this script
        - artifact path: latest run which produced this file
    """
    # Try as run_id
    if db.get_run(target):
        return target

    # Always resolve to absolute path
    resolved = str(Path(target).resolve())

    # Try as script path
    for run in db.list_runs(limit=100):
        if run.get("script_path") == resolved:
            return run["session_id"]

    # Try as artifact (output) path
    sessions = db.find_session_by_file(resolved, role="output")
    return sessions[0] if sessions else None


def _unknown_result(session_id: str, script_path: str) -> RunVerification:
    """Create an unknown verification result."""
    return RunVerification(
        session_id=session_id,
        script_path=script_path,
        status=VerificationStatus.UNKNOWN,
        files=[],
        combined_hash_expected=None,
        combined_hash_current=None,
        level=VerificationLevel.RERUN,
    )


def _execute_script(script_path: str, timeout: int) -> RunVerification | None:
    """Execute script and return error result if failed, None if success."""
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            timeout=timeout,
            cwd=Path(script_path).parent,
        )
        if result.returncode != 0:
            return RunVerification(
                session_id="",
                script_path=script_path,
                status=VerificationStatus.MISMATCH,
                files=[],
                combined_hash_expected=None,
                combined_hash_current=None,
                level=VerificationLevel.RERUN,
            )
        return None  # Success
    except subprocess.TimeoutExpired:
        return _unknown_result("", script_path)
    except Exception:
        return _unknown_result("", script_path)


def _find_new_session(db, script_path: str, original_id: str) -> tuple:
    """Find the new session created by re-running the script."""
    recent_runs = db.list_runs(limit=5)
    for run in recent_runs:
        if run.get("script_path") == script_path and run["session_id"] != original_id:
            return run["session_id"], run.get("sdir_run")
    return None, None


def _compare_hashes(
    original_hashes: Dict[str, str], new_hashes: Dict[str, str]
) -> list:
    """Compare hashes by filename and return FileVerification list."""
    original_by_name = {Path(p).name: h for p, h in original_hashes.items()}
    new_by_name = {Path(p).name: h for p, h in new_hashes.items()}

    verifications = []
    for filename, expected_hash in original_by_name.items():
        current_hash = new_by_name.get(filename)
        if current_hash is None:
            status = VerificationStatus.MISSING
        elif current_hash == expected_hash:
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.MISMATCH

        verifications.append(
            FileVerification(
                path=filename,
                role="output",
                expected_hash=expected_hash,
                current_hash=current_hash,
                status=status,
            )
        )
    return verifications


def _cleanup_session_dir(sdir_run: str) -> None:
    """Remove the session's output directory (best-effort)."""
    try:
        path = Path(sdir_run)
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


def _determine_status(file_verifications: list) -> VerificationStatus:
    """Determine overall verification status from file verifications."""
    if all(f.is_verified for f in file_verifications):
        return VerificationStatus.VERIFIED
    if any(f.status == VerificationStatus.MISMATCH for f in file_verifications):
        return VerificationStatus.MISMATCH
    return VerificationStatus.UNKNOWN


# Backward compatibility alias
verify_run_from_scratch = verify_by_rerun


# EOF
