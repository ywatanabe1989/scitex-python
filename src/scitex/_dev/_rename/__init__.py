#!/usr/bin/env python3
# Timestamp: 2026-02-14
# File: scitex/_dev/_rename/__init__.py

"""Bulk rename utility for files, contents, directories, and symlinks.

Ported from rename.sh - Django-safe bulk renaming with two-level filtering.

Execution order (critical for path integrity):
    0. File contents    - Safe: doesn't change paths
    1. Symlink targets  - Update to future paths (before renaming files/dirs)
    2. Symlink names    - Rename symlink names (leaf nodes)
    3. File names       - Rename files (leaf nodes)
    4. Directory names  - Rename directories (deepest first, children -> parents)
"""

from ._config import RenameConfig, RenameResult
from ._core import bulk_rename, execute_rename, preview_rename

__all__ = [
    "RenameConfig",
    "RenameResult",
    "bulk_rename",
    "execute_rename",
    "preview_rename",
]

# EOF
