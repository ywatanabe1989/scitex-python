#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/types.py

"""
Git operation result types.

This module provides dataclasses for rich error handling and result tracking
in git operations, addressing the "boolean blindness" issue identified in EVAL.md.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GitResult:
    """
    Result of a git operation.

    Attributes
    ----------
    success : bool
        Whether the operation succeeded
    message : Optional[str]
        Human-readable message about the operation
    stdout : Optional[str]
        Standard output from git command
    stderr : Optional[str]
        Standard error from git command
    exit_code : Optional[int]
        Exit code from git command

    Examples
    --------
    >>> result = GitResult(success=True, message="Commit created")
    >>> if result.success:
    ...     print(f"Success: {result.message}")

    >>> result = GitResult(
    ...     success=False,
    ...     message="Not a git repository",
    ...     stderr="fatal: not a git repository"
    ... )
    >>> if not result.success:
    ...     print(f"Error: {result.message}")
    ...     print(f"Details: {result.stderr}")
    """

    success: bool
    message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None


@dataclass
class CommitResult(GitResult):
    """
    Result of a git commit operation.

    Attributes
    ----------
    commit_hash : Optional[str]
        SHA hash of the created commit

    Examples
    --------
    >>> result = CommitResult(
    ...     success=True,
    ...     message="Commit created",
    ...     commit_hash="abc123def456"
    ... )
    >>> if result.success:
    ...     print(f"Created commit {result.commit_hash}")
    """

    commit_hash: Optional[str] = None


@dataclass
class CloneResult(GitResult):
    """
    Result of a git clone operation.

    Attributes
    ----------
    repo_path : Optional[str]
        Path to the cloned repository

    Examples
    --------
    >>> result = CloneResult(
    ...     success=True,
    ...     message="Repository cloned",
    ...     repo_path="/path/to/repo"
    ... )
    >>> if result.success:
    ...     print(f"Cloned to {result.repo_path}")
    """

    repo_path: Optional[str] = None


@dataclass
class BranchResult(GitResult):
    """
    Result of a git branch operation.

    Attributes
    ----------
    branch_name : Optional[str]
        Name of the branch created or modified

    Examples
    --------
    >>> result = BranchResult(
    ...     success=True,
    ...     message="Branch created",
    ...     branch_name="feature/new-feature"
    ... )
    >>> if result.success:
    ...     print(f"Branch: {result.branch_name}")
    """

    branch_name: Optional[str] = None


__all__ = [
    "GitResult",
    "CommitResult",
    "CloneResult",
    "BranchResult",
]

# EOF
