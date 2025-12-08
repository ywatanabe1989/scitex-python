#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/validation.py

"""
Input validation utilities for git operations.
"""

import re
from pathlib import Path
from typing import Tuple


def validate_branch_name(name: str) -> Tuple[bool, str]:
    """
    Validate git branch name according to git naming rules.

    Parameters
    ----------
    name : str
        Branch name to validate

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not name or not name.strip():
        return False, "Branch name cannot be empty"

    name = name.strip()

    if name.startswith("-"):
        return False, "Branch name cannot start with '-'"

    if name.endswith(".lock"):
        return False, "Branch name cannot end with '.lock'"

    if ".." in name:
        return False, "Branch name cannot contain '..'"

    invalid_chars = ["~", "^", ":", "?", "*", "[", "\\", " ", "\t"]
    for char in invalid_chars:
        if char in name:
            return False, f"Branch name cannot contain '{char}'"

    if name.endswith("/"):
        return False, "Branch name cannot end with '/'"

    if name.startswith("/"):
        return False, "Branch name cannot start with '/'"

    if "//" in name:
        return False, "Branch name cannot contain consecutive slashes"

    return True, ""


def validate_commit_message(message: str) -> Tuple[bool, str]:
    """
    Validate git commit message.

    Parameters
    ----------
    message : str
        Commit message to validate

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not message or not message.strip():
        return False, "Commit message cannot be empty"

    return True, ""


def validate_path(path: Path, must_exist: bool = False) -> Tuple[bool, str]:
    """
    Validate path for git operations.

    Parameters
    ----------
    path : Path
        Path to validate
    must_exist : bool
        Whether path must exist

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    try:
        resolved = path.resolve()

        if must_exist and not resolved.exists():
            return False, f"Path does not exist: {path}"

        path_str = str(resolved)
        if ".." in path.parts:
            return False, "Path contains parent directory references"

        return True, ""

    except (OSError, RuntimeError) as e:
        return False, f"Invalid path: {e}"


__all__ = [
    "validate_branch_name",
    "validate_commit_message",
    "validate_path",
]

# EOF
