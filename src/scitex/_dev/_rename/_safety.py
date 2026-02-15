#!/usr/bin/env python3
# Timestamp: 2026-02-14
# File: scitex/_dev/_rename/_safety.py

"""Safety checks for bulk rename operations."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def has_uncommitted_changes(directory: str) -> bool:
    """Check if git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def is_git_repo(directory: str) -> bool:
    """Check if directory is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_directory_safety(directory: str) -> str | None:
    """Validate directory is safe for bulk rename.

    Returns None if safe, or an error message string.
    """
    resolved = Path(directory).resolve()

    # Block system-critical paths
    dangerous = {
        "/",
        "/home",
        "/usr",
        "/etc",
        "/var",
        "/bin",
        "/sbin",
        "/opt",
        "/tmp",
    }
    if str(resolved) in dangerous:
        return f"Refusing to rename in system directory: {resolved}"

    # Block shallow paths (less than 3 components like /home/user)
    if len(resolved.parts) < 3:
        return f"Refusing to rename in shallow directory: {resolved}"

    # Must be inside a git repo (so we can revert with git checkout)
    if not is_git_repo(str(resolved)):
        return (
            f"Directory is not inside a git repository: {resolved}. "
            "Rename requires git for safety (allows git checkout to revert)."
        )

    return None


def create_backup(directory: str, pattern: str, replacement: str) -> Path:
    """Create a backup of the directory before renaming."""
    backup_base = Path(directory) / ".rename_backups"
    backup_base.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_base / f"backup_{timestamp}"
    backup_dir.mkdir()

    # Save operation metadata
    meta = backup_dir / "operation.txt"
    meta.write_text(
        f"pattern={pattern}\nreplacement={replacement}\n"
        f"directory={directory}\ntimestamp={timestamp}\n"
    )

    # Copy directory contents
    original_dir = backup_dir / "original"
    original_dir.mkdir()
    for item in Path(directory).iterdir():
        if item.name == ".rename_backups":
            continue
        dest = original_dir / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(dest), symlinks=True)
        else:
            shutil.copy2(str(item), str(dest))

    return backup_dir


# EOF
